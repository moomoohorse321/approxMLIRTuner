// RUN: cgeist -O0 %stdinclude %s -S > %s.mlir
// RUN: cgeist -O0 %stdinclude %s -o -lm %s.exec

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>


#define fp double

#define NUMBER_PAR_PER_BOX 128							// keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

//#define NUMBER_THREADS 128								// this should be roughly equal to NUMBER_PAR_PER_BOX for best performance
// Parameterized work group size
#ifdef RD_WG_SIZE_0_0
        #define NUMBER_THREADS RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define NUMBER_THREADS RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define NUMBER_THREADS RD_WG_SIZE
#else
        #define NUMBER_THREADS 128
#endif

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))	// STABLE


typedef struct
{
	fp x, y, z;

} THREE_VECTOR;

typedef struct
{
	fp v, x, y, z;

} FOUR_VECTOR;

typedef struct nei_str
{

	// neighbor box
	int x, y, z;
	int number;
	long offset;

} nei_str;

typedef struct box_str
{

	// home box
	int x, y, z;
	int number;
	long offset;

	// neighbor boxes
	int nn;
	nei_str nei[26];

} box_str;

typedef struct par_str
{

	fp alpha;

} par_str;

typedef struct dim_str
{

	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;

	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;

} dim_str;


// -------------------- utilities (no knobs) --------------------
static long long get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec;
}
static int isInteger(const char *s){ if(!s||!*s) return 0; for(;*s;++s){ if(*s<'0'||*s>'9') return 0; } return 1; }

// -------------------- helper to build states (caller-side only) --------------------
static inline int state_pair_from_u2(fp u2){
    // Quantize u² into [0, ...], scaled by 100 (used by func_substitute threshold at 70)
    if (u2 < (fp)0) u2 = (fp)0;
    fp scaled = (fp)100.0 * u2;
    int s = (int)(scaled + (fp)0.5);
    return s;
}
static inline int state_self_from_qi(fp qi){
    // |q_i| in ~[0.1,1.0] -> [10,100]
    if (qi < (fp)0) qi = -qi;
    fp scaled = (fp)100.0 * qi;
    if (scaled > (fp)100.0) scaled = (fp)100.0;
    int s = (int)(scaled + (fp)0.5);
    return s;
}
static inline int state_neigh_from_nn(int nn){
    // neighbor count 0..26
    if (nn < 0) nn = 0; if (nn > 26) nn = 26;
    return nn;
}

// -------------------- pair interaction (func_substitute knob) --------------------


int pair_interaction(int pi, int pj, fp a2,
                             FOUR_VECTOR* rv, fp* qv,
                             FOUR_VECTOR* fv_particle, int state){
    fp r2 = rv[pi].v + rv[pj].v - DOT(rv[pi], rv[pj]);
    if (r2 < (fp)0) r2 = (fp)0;
    fp u2  = a2 * r2;
    fp vij = (fp)exp(-u2);
    fp fs  = (fp)2.0 * vij;

    THREE_VECTOR d;
    d.x = rv[pi].x - rv[pj].x;
    d.y = rv[pi].y - rv[pj].y;
    d.z = rv[pi].z - rv[pj].z;

    fv_particle->v += qv[pj] * vij;
    fv_particle->x += qv[pj] * (fs * d.x);
    fv_particle->y += qv[pj] * (fs * d.y);
    fv_particle->z += qv[pj] * (fs * d.z);
    return state; // not used in callee
}

int approx_pair_interaction(int pi, int pj, fp a2,
                                    FOUR_VECTOR* rv, fp* qv,
                                    FOUR_VECTOR* fv_particle, int state){
    // exp(-u²) ≈ 1/(1+u²)
    fp r2 = rv[pi].v + rv[pj].v - DOT(rv[pi], rv[pj]);
    if (r2 < (fp)0) r2 = (fp)0; // numerical safety
    fp u2 = a2 * r2;
    fp vij = (fp)1.0 / ((fp)1.0 + u2);
    fp fs  = (fp)2.0 * vij;

    THREE_VECTOR d;
    d.x = rv[pi].x - rv[pj].x;
    d.y = rv[pi].y - rv[pj].y;
    d.z = rv[pi].z - rv[pj].z;

    fv_particle->v += qv[pj] * vij;
    fv_particle->x += qv[pj] * (fs * d.x);
    fv_particle->y += qv[pj] * (fs * d.y);
    fv_particle->z += qv[pj] * (fs * d.z);
    return state; // not used in callee
}

// -------------------- self-box accumulate (loop_perforate knob) --------------------
static void self_box_accumulate(int pi_idx, int first_i, fp a2,
                                FOUR_VECTOR* rv, fp* qv,
                                FOUR_VECTOR* fv_particle, int state){
    // NOTE: 'state' belongs to this knob and is NOT used to alter behavior here.
    // We still compute a per-pair state for the nested knob (pair_interaction).
    int j;
    for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
        int pj_idx = first_i + j;
        if (pi_idx == pj_idx) continue;

        // caller-side state for nested knob A (based on u²)
        fp r2 = rv[pi_idx].v + rv[pj_idx].v - DOT(rv[pi_idx], rv[pj_idx]);
        if (r2 < (fp)0) r2 = (fp)0;
        fp u2 = a2 * r2;
        int pair_state = state_pair_from_u2(u2);

        pair_interaction(pi_idx, pj_idx, a2, rv, qv, fv_particle, pair_state);
    }
    (void)state;
}

// -------------------- neighbor-box accumulate (loop_perforate knob) --------------------
static void neighbor_box_accumulate(int pi_idx, int bx, fp a2,
                                    box_str* box, FOUR_VECTOR* rv, fp* qv,
                                    FOUR_VECTOR* fv_particle, int state){
    int k;
    for (k = 0; k < box[bx].nn; k++) {
        int nb  = box[bx].nei[k].number;
        int off = box[nb].offset;

        int j;
        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            int pj_idx = off + j;

            // caller-side state for nested knob A (based on u²)
            fp r2 = rv[pi_idx].v + rv[pj_idx].v - DOT(rv[pi_idx], rv[pj_idx]);
            if (r2 < (fp)0) r2 = (fp)0;
            fp u2 = a2 * r2;
            int pair_state = state_pair_from_u2(u2);

            pair_interaction(pi_idx, pj_idx, a2, rv, qv, fv_particle, pair_state);
        }
    }
    (void)state;
}

// -------------------- non-knob wrapper over a box --------------------
static void process_home_box(int bx, fp a2, box_str* box,
                             FOUR_VECTOR* rv, fp* qv, FOUR_VECTOR* fv){
    int first_i = box[bx].offset;
    int i;
    for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {
        int pi_idx = first_i + i;

        FOUR_VECTOR acc = {0,0,0,0};

        // Build states for the two outer knobs (caller-side!)
        int state_self  = state_self_from_qi(qv[pi_idx]);          // for self_box_accumulate
        int state_neigh = state_neigh_from_nn(box[bx].nn);         // for neighbor_box_accumulate

        self_box_accumulate   (pi_idx, first_i, a2, rv, qv, &acc, state_self);
        neighbor_box_accumulate(pi_idx, bx,      a2, box, rv, qv, &acc, state_neigh);

        fv[pi_idx].v += acc.v;
        fv[pi_idx].x += acc.x;
        fv[pi_idx].y += acc.y;
        fv[pi_idx].z += acc.z;
    }
}

// -------------------- main (unchanged I/O + build) --------------------
int main(int argc, char *argv[]) {
    par_str par_cpu;
    dim_str dim_cpu;
    box_str*     box_cpu = NULL;
    FOUR_VECTOR* rv_cpu  = NULL;
    fp*          qv_cpu  = NULL;
    FOUR_VECTOR* fv_cpu  = NULL;
    int nh;

    printf("WG size of kernel = %d\n", NUMBER_THREADS);

    dim_cpu.boxes1d_arg = 1;
    if (argc == 3) {
        if (strcmp(argv[1], "-boxes1d") == 0 && isInteger(argv[2])) {
            dim_cpu.boxes1d_arg = atoi(argv[2]);
            if (dim_cpu.boxes1d_arg <= 0) { printf("ERROR: -boxes1d > 0\n"); return 1; }
        } else { printf("ERROR: Usage: %s -boxes1d <number>\n", argv[0]); return 1; }
    } else { printf("Usage: %s -boxes1d <number>\n", argv[0]); return 1; }

    printf("Configuration: boxes1d = %d\n", dim_cpu.boxes1d_arg);

    par_cpu.alpha        = (fp)0.5;
    dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;
    dim_cpu.space_elem   = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;

    box_cpu = (box_str*)malloc((size_t)dim_cpu.number_boxes * sizeof(box_str));
    rv_cpu  = (FOUR_VECTOR*)malloc((size_t)dim_cpu.space_elem   * sizeof(FOUR_VECTOR));
    qv_cpu  = (fp*)malloc((size_t)dim_cpu.space_elem            * sizeof(fp));
    fv_cpu  = (FOUR_VECTOR*)malloc((size_t)dim_cpu.space_elem   * sizeof(FOUR_VECTOR));
    if (!box_cpu || !rv_cpu || !qv_cpu || !fv_cpu) { printf("ERROR: OOM\n"); free(rv_cpu); free(qv_cpu); free(fv_cpu); free(box_cpu); return 1; }

    // build neighbors
    nh = 0;
    for (int i=0;i<dim_cpu.boxes1d_arg;i++){
      for (int j=0;j<dim_cpu.boxes1d_arg;j++){
        for (int k=0;k<dim_cpu.boxes1d_arg;k++){
          box_cpu[nh].number = nh;
          box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;
          box_cpu[nh].nn = 0;
          for (int l=-1;l<=1;l++) for (int m=-1;m<=1;m++) for (int n=-1;n<=1;n++){
            if (!(l==0 && m==0 && n==0)){
              int ni=i+l,nj=j+m,nk=k+n;
              if (ni>=0 && ni<dim_cpu.boxes1d_arg && nj>=0 && nj<dim_cpu.boxes1d_arg && nk>=0 && nk<dim_cpu.boxes1d_arg){
                int idx = box_cpu[nh].nn++;
                int num = (ni*dim_cpu.boxes1d_arg*dim_cpu.boxes1d_arg) + (nj*dim_cpu.boxes1d_arg) + nk;
                box_cpu[nh].nei[idx].number = num;
                box_cpu[nh].nei[idx].offset = num * NUMBER_PAR_PER_BOX;
              }
            }
          }
          nh++;
        }
      }
    }

    // init fields
    srand(2);
    for (size_t i=0;i<(size_t)dim_cpu.space_elem;i++){
      rv_cpu[i].v=(fp)((rand()%10+1)/10.0);
      rv_cpu[i].x=(fp)((rand()%10+1)/10.0);
      rv_cpu[i].y=(fp)((rand()%10+1)/10.0);
      rv_cpu[i].z=(fp)((rand()%10+1)/10.0);
      qv_cpu[i]  =(fp)((rand()%10+1)/10.0);
      fv_cpu[i].v=fv_cpu[i].x=fv_cpu[i].y=fv_cpu[i].z=(fp)0.0;
    }

    // run
    long long t0=get_time();
    fp a2 = (fp)2.0 * par_cpu.alpha * par_cpu.alpha;
    for (int bx=0; bx<dim_cpu.number_boxes; ++bx) {
        process_home_box(bx, a2, box_cpu, rv_cpu, qv_cpu, fv_cpu);
    }
    long long t1=get_time();
    printf("Total execution time: %f seconds\n", (double)(t1-t0)/1e6);

    printf("\n--- Simulation Statistics ---\n");
    printf("Total Particles: %ld\n", dim_cpu.space_elem);
    printf("Number of Boxes: %ld\n", dim_cpu.number_boxes);
    printf("Particles per Box: %d\n", NUMBER_PAR_PER_BOX);

    fp v_sum = 0.0, x_sum = 0.0, y_sum = 0.0, z_sum = 0.0;
    fp v_max = -INFINITY, x_max = -INFINITY, y_max = -INFINITY, z_max = -INFINITY;
    fp v_min = INFINITY,  x_min = INFINITY,  y_min = INFINITY,  z_min = INFINITY;

    for (size_t i = 0; i < (size_t)dim_cpu.space_elem; i++) {
        v_sum += fv_cpu[i].v;
        x_sum += fv_cpu[i].x;
        y_sum += fv_cpu[i].y;
        z_sum += fv_cpu[i].z;

        if (fv_cpu[i].v > v_max) v_max = fv_cpu[i].v;
        if (fv_cpu[i].x > x_max) x_max = fv_cpu[i].x;
        if (fv_cpu[i].y > y_max) y_max = fv_cpu[i].y;
        if (fv_cpu[i].z > z_max) z_max = fv_cpu[i].z;
        
        if (fv_cpu[i].v < v_min) v_min = fv_cpu[i].v;
        if (fv_cpu[i].x < x_min) x_min = fv_cpu[i].x;
        if (fv_cpu[i].y < y_min) y_min = fv_cpu[i].y;
        if (fv_cpu[i].z < z_min) z_min = fv_cpu[i].z;
    }

    printf("\n--- Result Summary (fv_cpu) ---\n");
    printf("        Component |      Average |          Min |          Max\n");
    printf("------------------|--------------|--------------|--------------\n");
    printf("Potential Energy (v) | %12.6f | %12.6f | %12.6f\n", v_sum / dim_cpu.space_elem, v_min, v_max);
    printf("      Force Vector (x) | %12.6f | %12.6f | %12.6f\n", x_sum / dim_cpu.space_elem, x_min, x_max);
    printf("      Force Vector (y) | %12.6f | %12.6f | %12.6f\n", y_sum / dim_cpu.space_elem, y_min, y_max);
    printf("      Force Vector (z) | %12.6f | %12.6f | %12.6f\n", z_sum / dim_cpu.space_elem, z_min, z_max);


    // Print the full results to stdout instead of writing to a file
    printf("\n--- Full Particle Data Dump (fv_cpu) ---\n");
    printf("Particle Index | Potential (v) |   Force (x) |   Force (y) |   Force (z)\n");
    printf("---------------|---------------|-------------|-------------|-------------\n");
    for (size_t i = 0; i < (size_t)dim_cpu.space_elem; i++) {
        printf("%14zu | %13.6f | %11.6f | %11.6f | %11.6f\n",
               i, fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
    }

        // write results
    FILE* fp_out = fopen("result.txt","wb");
    if (fp_out){
      size_t wrote = fwrite(&dim_cpu.space_elem, sizeof(dim_cpu.space_elem), 1, fp_out);
      if (wrote!=1) printf("ERROR: header write\n");
      wrote = fwrite(fv_cpu, sizeof(FOUR_VECTOR), (size_t)dim_cpu.space_elem, fp_out);
      if (wrote!=(size_t)dim_cpu.space_elem) printf("ERROR: data write\n");
      fclose(fp_out);
      printf("Results written to result.txt\n");
    } else { printf("ERROR: open result.txt\n"); }


    free(rv_cpu); free(qv_cpu); free(fv_cpu); free(box_cpu);
    return 0;
}
