module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  // Knob A — in-neighbor accumulation (loop_perforate)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "compute_sum_in_neighbors",
    transform_type = "loop_perforate",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 10>,
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 1, 2, 3, 4>,
    thresholds = array<i32: 7>,
    decisions = array<i32: 1, 1>
  }> : () -> ()

  // Knob B — per-node update (func_substitute)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "update_node_rank",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 50>,
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 6>,
    decisions = array<i32: 0, 0>
  }> : () -> ()

  // Required by func_substitute
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "update_node_rank"}> : () -> ()


  // Knob C - pagerank_worker_impl
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "pagerank_worker_impl",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 5>,
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 1>,
    decisions = array<i32: 1, 0>
  }> : () -> ()

  // Required by func_substitute
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "pagerank_worker_impl"}> : () -> ()
  llvm.mlir.global internal constant @str39("calloc\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str38("realloc\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str37("malloc\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str36(" %d %d\00") {addr_space = 0 : i32}
  llvm.func @__isoc99_sscanf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str35("Failed to open '%s': %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str34("r\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str33("Usage: %s [options]\0A  -m, --mode MODE         'synthetic' (default) or 'file'\0A  -f, --file PATH         edge-list file (u v per line), required for mode=file\0A  -t, --threads P         number of threads (default: 1)\0A  -n, --nodes N           nodes for synthetic (default: 10000)\0A  -d, --degree D          ~in-degree per node for synthetic (default: 10)\0A  -i, --iters K           iterations (default: 50)\0A  -a, --alpha A           damping (default: 0.85)\0A  -s, --seed S            RNG seed for synthetic (default: 1)\0A  -p, --print             print final ranks (can be large!)\0A  -h, --help              show this help\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str32("pr(%d) = %.12f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str31("%4d | %9d | %.12f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str30("-----|-----------|------------------\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str29("Rank |   Node ID |   PageRank Score\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str28("\0A--- Top 20 Ranked Pages ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str27("    Max Rank: %.12f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str26("    Min Rank: %.12f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str25("Average Rank: %.12f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str24("   Sum of Ranks: %.6f (should be ~1.0)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str23("  Total Nodes: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str22("\0A--- PageRank Statistics ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str21("Time: %.6f seconds\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str20("pthread_create\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str19("pthread_barrier_init\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str18("Unknown mode '%s'\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("Synthetic graph: N=%d, ~in-degree=%d, M=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("Invalid N or DEG for synthetic graph.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("Loaded graph from '%s': N=%d, M=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stdout() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.mlir.global internal constant @str14("Empty or unreadable graph file.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("mode=file requires --file PATH\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("alpha must be in (0,1), got %g\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  memref.global @optarg : memref<1xmemref<?xi8>>
  llvm.mlir.global internal constant @str11("e:m:f:t:n:d:i:a:s:ph\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("help\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("print\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("seed\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("alpha\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("iters\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("degree\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("nodes\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("threads\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("file\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("mode\00") {addr_space = 0 : i32}
  memref.global "private" @"main@static@long_opts@init" : memref<1xi1> = dense<true>
  memref.global "private" @"main@static@long_opts" : memref<11x!llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>> = uninitialized
  llvm.mlir.global internal constant @str0("synthetic\00") {addr_space = 0 : i32}
  func.func @compute_sum_in_neighbors(%arg0: memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, %arg1: memref<?xf64>, %arg2: i32, %arg3: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %2 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %3 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = affine.load %3[symbol(%4)] : memref<?xi32>
    %6 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %7 = llvm.getelementptr %6[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %8 = llvm.load %7 : !llvm.ptr -> memref<?xi32>
    %9 = affine.load %8[symbol(%0) + 1] : memref<?xi32>
    %10 = arith.index_cast %9 : i32 to index
    %11 = arith.index_cast %5 : i32 to index
    %12 = scf.for %arg4 = %11 to %10 step %c1 iter_args(%arg5 = %cst) -> (f64) {
      %13 = arith.subi %arg4, %11 : index
      %14 = arith.index_cast %5 : i32 to index
      %15 = arith.addi %14, %13 : index
      %16 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %17 = llvm.getelementptr %16[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
      %18 = llvm.load %17 : !llvm.ptr -> memref<?xi32>
      %19 = memref.load %18[%15] : memref<?xi32>
      %20 = llvm.getelementptr %16[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
      %21 = llvm.load %20 : !llvm.ptr -> memref<?xi32>
      %22 = arith.index_cast %19 : i32 to index
      %23 = memref.load %21[%22] : memref<?xi32>
      %24 = arith.cmpi sgt, %23, %c0_i32 : i32
      %25 = scf.if %24 -> (f64) {
        %26 = memref.load %arg1[%22] : memref<?xf64>
        %27 = arith.sitofp %23 : i32 to f64
        %28 = arith.divf %26, %27 : f64
        %29 = arith.addf %arg5, %28 : f64
        scf.yield %29 : f64
      } else {
        scf.yield %arg5 : f64
      }
      scf.yield %25 : f64
    }
    return %12 : f64
  }
  func.func @approx_update_node_rank(%arg0: memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, %arg1: memref<?xf64>, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: i32, %arg7: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %2 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %3 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = affine.load %3[symbol(%4)] : memref<?xi32>
    %6 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %7 = llvm.getelementptr %6[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %8 = llvm.load %7 : !llvm.ptr -> memref<?xi32>
    %9 = affine.load %8[symbol(%0) + 1] : memref<?xi32>
    %10 = arith.subi %9, %5 : i32
    %11 = arith.cmpi sle, %10, %c2_i32 : i32
    %12 = arith.select %11, %c1_i32, %c2_i32 : i32
    %13 = arith.cmpi sgt, %12, %c1_i32 : i32
    %14 = arith.index_cast %9 : i32 to index
    %15 = arith.index_cast %5 : i32 to index
    %16 = arith.index_cast %12 : i32 to index
    %17 = scf.for %arg8 = %15 to %14 step %16 iter_args(%arg9 = %cst) -> (f64) {
      %22 = arith.subi %arg8, %15 : index
      %23 = arith.divui %22, %16 : index
      %24 = arith.index_cast %12 : i32 to index
      %25 = arith.muli %23, %24 : index
      %26 = arith.index_cast %5 : i32 to index
      %27 = arith.addi %26, %25 : index
      %28 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %29 = llvm.getelementptr %28[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
      %30 = llvm.load %29 : !llvm.ptr -> memref<?xi32>
      %31 = memref.load %30[%27] : memref<?xi32>
      %32 = llvm.getelementptr %28[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
      %33 = llvm.load %32 : !llvm.ptr -> memref<?xi32>
      %34 = arith.index_cast %31 : i32 to index
      %35 = memref.load %33[%34] : memref<?xi32>
      %36 = arith.cmpi sgt, %35, %c0_i32 : i32
      %37 = scf.if %36 -> (f64) {
        %38 = memref.load %arg1[%34] : memref<?xf64>
        %39 = arith.sitofp %35 : i32 to f64
        %40 = arith.divf %38, %39 : f64
        %41 = arith.addf %arg9, %40 : f64
        scf.yield %41 : f64
      } else {
        scf.yield %arg9 : f64
      }
      scf.yield %37 : f64
    }
    %18 = scf.if %13 -> (f64) {
      %22 = arith.sitofp %12 : i32 to f64
      %23 = arith.mulf %17, %22 : f64
      scf.yield %23 : f64
    } else {
      scf.yield %17 : f64
    }
    %19 = arith.addf %arg4, %arg5 : f64
    %20 = arith.mulf %arg3, %18 : f64
    %21 = arith.addf %19, %20 : f64
    return %21 : f64
  }
  func.func @update_node_rank(%arg0: memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, %arg1: memref<?xf64>, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: i32, %arg7: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = call @compute_sum_in_neighbors(%arg0, %arg1, %arg2, %arg6) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, i32, i32) -> f64
    %1 = arith.addf %arg4, %arg5 : f64
    %2 = arith.mulf %arg3, %0 : f64
    %3 = arith.addf %1, %2 : f64
    return %3 : f64
  }
  func.func @approx_pagerank_worker_impl(%arg0: memref<?xi8>, %arg1: i32) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %2 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = llvm.getelementptr %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %6 = llvm.load %5 : !llvm.ptr -> i32
    %7 = llvm.getelementptr %2[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %8 = llvm.load %7 : !llvm.ptr -> i32
    %9 = arith.cmpi eq, %3, %c0_i32 : i32
    %10 = llvm.getelementptr %2[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %11 = llvm.load %10 : !llvm.ptr -> memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>
    %12 = llvm.getelementptr %2[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.getelementptr %2[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %15 = llvm.load %14 : !llvm.ptr -> f64
    %16 = arith.muli %3, %8 : i32
    %17 = arith.divsi %16, %6 : i32
    %18 = arith.addi %3, %c1_i32 : i32
    %19 = arith.muli %18, %8 : i32
    %20 = arith.divsi %19, %6 : i32
    %21 = llvm.getelementptr %2[0, 12] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %22 = arith.index_cast %20 : i32 to index
    %23 = arith.index_cast %17 : i32 to index
    %24 = llvm.getelementptr %2[0, 9] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %25 = llvm.getelementptr %2[0, 10] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %26 = arith.index_cast %20 : i32 to index
    %27 = arith.index_cast %17 : i32 to index
    %28 = arith.index_cast %20 : i32 to index
    %29 = arith.index_cast %17 : i32 to index
    %30:3 = scf.while (%arg2 = %0, %arg3 = %1, %arg4 = %c0_i32) : (i32, f64, i32) -> (i32, f64, i32) {
      %33 = llvm.load %21 : !llvm.ptr -> i32
      %34 = arith.cmpi slt, %arg4, %33 : i32
      scf.condition(%34) %arg2, %arg3, %arg4 : i32, f64, i32
    } do {
    ^bb0(%arg2: i32, %arg3: f64, %arg4: i32):
      %33 = scf.for %arg5 = %23 to %22 step %c1 iter_args(%arg6 = %cst) -> (f64) {
        %40 = arith.subi %arg5, %23 : index
        %41 = arith.index_cast %17 : i32 to index
        %42 = arith.addi %41, %40 : index
        %43 = "polygeist.memref2pointer"(%11) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %44 = llvm.getelementptr %43[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        %45 = llvm.load %44 : !llvm.ptr -> memref<?xi32>
        %46 = memref.load %45[%42] : memref<?xi32>
        %47 = arith.cmpi eq, %46, %c0_i32 : i32
        %48 = scf.if %47 -> (f64) {
          %49 = llvm.getelementptr %2[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
          %50 = llvm.load %49 : !llvm.ptr -> memref<?xf64>
          %51 = memref.load %50[%42] : memref<?xf64>
          %52 = arith.addf %arg6, %51 : f64
          scf.yield %52 : f64
        } else {
          scf.yield %arg6 : f64
        }
        scf.yield %48 : f64
      }
      %34 = llvm.load %24 : !llvm.ptr -> memref<?xf64>
      affine.store %33, %34[symbol(%4)] : memref<?xf64>
      %35 = arith.select %9, %6, %arg2 : i32
      %36 = scf.if %9 -> (f64) {
        %40 = arith.index_cast %6 : i32 to index
        %41 = scf.for %arg5 = %c0 to %40 step %c1 iter_args(%arg6 = %cst) -> (f64) {
          %47 = llvm.getelementptr %2[0, 9] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
          %48 = llvm.load %47 : !llvm.ptr -> memref<?xf64>
          %49 = memref.load %48[%arg5] : memref<?xf64>
          %50 = arith.addf %arg6, %49 : f64
          scf.yield %50 : f64
        }
        %42 = llvm.getelementptr %2[0, 10] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %43 = llvm.load %42 : !llvm.ptr -> memref<?xf64>
        %44 = arith.sitofp %8 : i32 to f64
        %45 = arith.divf %41, %44 : f64
        %46 = arith.mulf %13, %45 : f64
        affine.store %46, %43[0] : memref<?xf64>
        scf.yield %41 : f64
      } else {
        scf.yield %arg3 : f64
      }
      %37 = llvm.load %25 : !llvm.ptr -> memref<?xf64>
      %38 = affine.load %37[0] : memref<?xf64>
      scf.for %arg5 = %27 to %26 step %c1 {
        %40 = arith.subi %arg5, %27 : index
        %41 = arith.index_cast %17 : i32 to index
        %42 = arith.addi %41, %40 : index
        %43 = arith.index_cast %42 : index to i32
        %44 = "polygeist.memref2pointer"(%11) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %45 = llvm.getelementptr %44[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        %46 = llvm.load %45 : !llvm.ptr -> memref<?xi32>
        %47 = arith.addi %43, %c1_i32 : i32
        %48 = arith.index_cast %47 : i32 to index
        %49 = memref.load %46[%48] : memref<?xi32>
        %50 = memref.load %46[%42] : memref<?xi32>
        %51 = arith.subi %49, %50 : i32
        %52 = llvm.getelementptr %2[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %53 = llvm.load %52 : !llvm.ptr -> memref<?xf64>
        %54 = llvm.getelementptr %2[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %55 = llvm.load %54 : !llvm.ptr -> memref<?xf64>
        %56 = func.call @update_node_rank(%11, %55, %43, %13, %15, %38, %51, %arg4) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, i32, f64, f64, f64, i32, i32) -> f64
        memref.store %56, %53[%42] : memref<?xf64>
      }
      scf.for %arg5 = %29 to %28 step %c1 {
        %40 = arith.subi %arg5, %29 : index
        %41 = arith.index_cast %17 : i32 to index
        %42 = arith.addi %41, %40 : index
        %43 = llvm.getelementptr %2[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %44 = llvm.load %43 : !llvm.ptr -> memref<?xf64>
        %45 = llvm.getelementptr %2[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %46 = llvm.load %45 : !llvm.ptr -> memref<?xf64>
        %47 = memref.load %46[%42] : memref<?xf64>
        memref.store %47, %44[%42] : memref<?xf64>
      }
      %39 = arith.addi %arg4, %c2_i32 : i32
      scf.yield %35, %36, %39 : i32, f64, i32
    }
    %31 = llvm.mlir.zero : !llvm.ptr
    %32 = "polygeist.pointer2memref"(%31) : (!llvm.ptr) -> memref<?xi8>
    return %32 : memref<?xi8>
  }
  func.func @pagerank_worker_impl(%arg0: memref<?xi8>, %arg1: i32) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %2 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = llvm.getelementptr %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %6 = llvm.load %5 : !llvm.ptr -> i32
    %7 = llvm.getelementptr %2[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %8 = llvm.load %7 : !llvm.ptr -> i32
    %9 = arith.cmpi eq, %3, %c0_i32 : i32
    %10 = llvm.getelementptr %2[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %11 = llvm.load %10 : !llvm.ptr -> memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>
    %12 = llvm.getelementptr %2[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.getelementptr %2[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %15 = llvm.load %14 : !llvm.ptr -> f64
    %16 = arith.muli %3, %8 : i32
    %17 = arith.divsi %16, %6 : i32
    %18 = arith.addi %3, %c1_i32 : i32
    %19 = arith.muli %18, %8 : i32
    %20 = arith.divsi %19, %6 : i32
    %21 = llvm.getelementptr %2[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %22 = llvm.load %21 : !llvm.ptr -> memref<?x!llvm.struct<(i64, array<24 x i8>)>>
    %23 = call @pthread_barrier_wait(%22) : (memref<?x!llvm.struct<(i64, array<24 x i8>)>>) -> i32
    %24 = llvm.getelementptr %2[0, 12] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %25 = arith.index_cast %20 : i32 to index
    %26 = arith.index_cast %17 : i32 to index
    %27 = llvm.getelementptr %2[0, 9] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %28 = llvm.getelementptr %2[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %29 = llvm.getelementptr %2[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %30 = llvm.getelementptr %2[0, 10] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %31 = arith.index_cast %20 : i32 to index
    %32 = arith.index_cast %17 : i32 to index
    %33 = llvm.getelementptr %2[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %34 = arith.index_cast %20 : i32 to index
    %35 = arith.index_cast %17 : i32 to index
    %36 = llvm.getelementptr %2[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %37:3 = scf.while (%arg2 = %0, %arg3 = %1, %arg4 = %c0_i32) : (i32, f64, i32) -> (i32, f64, i32) {
      %40 = llvm.load %24 : !llvm.ptr -> i32
      %41 = arith.cmpi slt, %arg4, %40 : i32
      scf.condition(%41) %arg2, %arg3, %arg4 : i32, f64, i32
    } do {
    ^bb0(%arg2: i32, %arg3: f64, %arg4: i32):
      %40 = scf.for %arg5 = %26 to %25 step %c1 iter_args(%arg6 = %cst) -> (f64) {
        %55 = arith.subi %arg5, %26 : index
        %56 = arith.index_cast %17 : i32 to index
        %57 = arith.addi %56, %55 : index
        %58 = "polygeist.memref2pointer"(%11) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %59 = llvm.getelementptr %58[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        %60 = llvm.load %59 : !llvm.ptr -> memref<?xi32>
        %61 = memref.load %60[%57] : memref<?xi32>
        %62 = arith.cmpi eq, %61, %c0_i32 : i32
        %63 = scf.if %62 -> (f64) {
          %64 = llvm.getelementptr %2[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
          %65 = llvm.load %64 : !llvm.ptr -> memref<?xf64>
          %66 = memref.load %65[%57] : memref<?xf64>
          %67 = arith.addf %arg6, %66 : f64
          scf.yield %67 : f64
        } else {
          scf.yield %arg6 : f64
        }
        scf.yield %63 : f64
      }
      %41 = llvm.load %27 : !llvm.ptr -> memref<?xf64>
      affine.store %40, %41[symbol(%4)] : memref<?xf64>
      %42 = llvm.load %28 : !llvm.ptr -> memref<?x!llvm.struct<(i64, array<24 x i8>)>>
      %43 = func.call @pthread_barrier_wait(%42) : (memref<?x!llvm.struct<(i64, array<24 x i8>)>>) -> i32
      %44 = arith.select %9, %6, %arg2 : i32
      %45 = scf.if %9 -> (f64) {
        %55 = arith.index_cast %6 : i32 to index
        %56 = scf.for %arg5 = %c0 to %55 step %c1 iter_args(%arg6 = %cst) -> (f64) {
          %62 = llvm.getelementptr %2[0, 9] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
          %63 = llvm.load %62 : !llvm.ptr -> memref<?xf64>
          %64 = memref.load %63[%arg5] : memref<?xf64>
          %65 = arith.addf %arg6, %64 : f64
          scf.yield %65 : f64
        }
        %57 = llvm.getelementptr %2[0, 10] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %58 = llvm.load %57 : !llvm.ptr -> memref<?xf64>
        %59 = arith.sitofp %8 : i32 to f64
        %60 = arith.divf %56, %59 : f64
        %61 = arith.mulf %13, %60 : f64
        affine.store %61, %58[0] : memref<?xf64>
        scf.yield %56 : f64
      } else {
        scf.yield %arg3 : f64
      }
      %46 = llvm.load %29 : !llvm.ptr -> memref<?x!llvm.struct<(i64, array<24 x i8>)>>
      %47 = func.call @pthread_barrier_wait(%46) : (memref<?x!llvm.struct<(i64, array<24 x i8>)>>) -> i32
      %48 = llvm.load %30 : !llvm.ptr -> memref<?xf64>
      %49 = affine.load %48[0] : memref<?xf64>
      scf.for %arg5 = %32 to %31 step %c1 {
        %55 = arith.subi %arg5, %32 : index
        %56 = arith.index_cast %17 : i32 to index
        %57 = arith.addi %56, %55 : index
        %58 = arith.index_cast %57 : index to i32
        %59 = "polygeist.memref2pointer"(%11) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %60 = llvm.getelementptr %59[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        %61 = llvm.load %60 : !llvm.ptr -> memref<?xi32>
        %62 = arith.addi %58, %c1_i32 : i32
        %63 = arith.index_cast %62 : i32 to index
        %64 = memref.load %61[%63] : memref<?xi32>
        %65 = memref.load %61[%57] : memref<?xi32>
        %66 = arith.subi %64, %65 : i32
        %67 = llvm.getelementptr %2[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %68 = llvm.load %67 : !llvm.ptr -> memref<?xf64>
        %69 = llvm.getelementptr %2[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %70 = llvm.load %69 : !llvm.ptr -> memref<?xf64>
        %71 = func.call @update_node_rank(%11, %70, %58, %13, %15, %49, %66, %arg4) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, i32, f64, f64, f64, i32, i32) -> f64
        memref.store %71, %68[%57] : memref<?xf64>
      }
      %50 = llvm.load %33 : !llvm.ptr -> memref<?x!llvm.struct<(i64, array<24 x i8>)>>
      %51 = func.call @pthread_barrier_wait(%50) : (memref<?x!llvm.struct<(i64, array<24 x i8>)>>) -> i32
      scf.for %arg5 = %35 to %34 step %c1 {
        %55 = arith.subi %arg5, %35 : index
        %56 = arith.index_cast %17 : i32 to index
        %57 = arith.addi %56, %55 : index
        %58 = llvm.getelementptr %2[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %59 = llvm.load %58 : !llvm.ptr -> memref<?xf64>
        %60 = llvm.getelementptr %2[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %61 = llvm.load %60 : !llvm.ptr -> memref<?xf64>
        %62 = memref.load %61[%57] : memref<?xf64>
        memref.store %62, %59[%57] : memref<?xf64>
      }
      %52 = llvm.load %36 : !llvm.ptr -> memref<?x!llvm.struct<(i64, array<24 x i8>)>>
      %53 = func.call @pthread_barrier_wait(%52) : (memref<?x!llvm.struct<(i64, array<24 x i8>)>>) -> i32
      %54 = arith.addi %arg4, %c1_i32 : i32
      scf.yield %44, %45, %54 : i32, f64, i32
    }
    %38 = llvm.mlir.zero : !llvm.ptr
    %39 = "polygeist.pointer2memref"(%38) : (!llvm.ptr) -> memref<?xi8>
    return %39 : memref<?xi8>
  }
  func.func private @pthread_barrier_wait(memref<?x!llvm.struct<(i64, array<24 x i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @pagerank_worker(%arg0: memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %1 = llvm.getelementptr %0[0, 13] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
    %2 = llvm.load %1 : !llvm.ptr -> i32
    %3 = call @pagerank_worker_impl(%arg0, %2) : (memref<?xi8>, i32) -> memref<?xi8>
    return %3 : memref<?xi8>
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c15_i64 = arith.constant 15 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c20_i32 = arith.constant 20 : i32
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c-1_i32 = arith.constant -1 : i32
    %c104_i32 = arith.constant 104 : i32
    %c112_i32 = arith.constant 112 : i32
    %c115_i32 = arith.constant 115 : i32
    %c97_i32 = arith.constant 97 : i32
    %c105_i32 = arith.constant 105 : i32
    %c100_i32 = arith.constant 100 : i32
    %c110_i32 = arith.constant 110 : i32
    %c116_i32 = arith.constant 116 : i32
    %c102_i32 = arith.constant 102 : i32
    %c109_i32 = arith.constant 109 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 8.500000e-01 : f64
    %c50_i32 = arith.constant 50 : i32
    %c10_i32 = arith.constant 10 : i32
    %c10000_i32 = arith.constant 10000 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16 = arith.constant 16 : index
    %c0_i8 = arith.constant 0 : i8
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %alloca = memref.alloca() : memref<1xf64>
    affine.store %1, %alloca[0] : memref<1xf64>
    %alloca_2 = memref.alloca() : memref<1x!llvm.struct<(i64, array<24 x i8>)>>
    %alloca_3 = memref.alloca() : memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>
    %alloca_4 = memref.alloca() : memref<1xi32>
    affine.store %0, %alloca_4[0] : memref<1xi32>
    %2 = memref.get_global @"main@static@long_opts" : memref<11x!llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>>
    %alloca_5 = memref.alloca() : memref<16xi8>
    affine.store %c0_i8, %alloca_5[0] : memref<16xi8>
    %cast = memref.cast %alloca_5 : memref<16xi8> to memref<?xi8>
    %3 = llvm.mlir.addressof @str0 : !llvm.ptr
    %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr) -> memref<?xi8>
    %5 = call @strncpy(%cast, %4, %c15_i64) : (memref<?xi8>, memref<?xi8>, i64) -> memref<?xi8>
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr) -> memref<?xi8>
    %8 = memref.get_global @"main@static@long_opts@init" : memref<1xi1>
    %9 = affine.load %8[0] : memref<1xi1>
    scf.if %9 {
      affine.store %false, %8[0] : memref<1xi1>
      %16 = "polygeist.memref2pointer"(%2) : (memref<11x!llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>>) -> !llvm.ptr
      %17 = llvm.getelementptr %16[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c1_i32, %17 : i32, !llvm.ptr
      %18 = llvm.getelementptr %16[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      %19 = llvm.mlir.zero : !llvm.ptr
      %20 = "polygeist.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi32>
      llvm.store %20, %18 : memref<?xi32>, !llvm.ptr
      %21 = llvm.getelementptr %16[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c109_i32, %21 : i32, !llvm.ptr
      %22 = "polygeist.typeSize"() <{source = !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>}> : () -> index
      %23 = arith.index_cast %22 : index to i64
      %24 = llvm.getelementptr %16[%23] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %25 = llvm.getelementptr %24[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c1_i32, %25 : i32, !llvm.ptr
      %26 = llvm.getelementptr %24[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %26 : memref<?xi32>, !llvm.ptr
      %27 = llvm.getelementptr %24[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c102_i32, %27 : i32, !llvm.ptr
      %28 = arith.muli %22, %c2 : index
      %29 = arith.index_cast %28 : index to i64
      %30 = llvm.getelementptr %16[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %31 = llvm.getelementptr %30[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c1_i32, %31 : i32, !llvm.ptr
      %32 = llvm.getelementptr %30[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %32 : memref<?xi32>, !llvm.ptr
      %33 = llvm.getelementptr %30[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c116_i32, %33 : i32, !llvm.ptr
      %34 = arith.muli %22, %c3 : index
      %35 = arith.index_cast %34 : index to i64
      %36 = llvm.getelementptr %16[%35] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %37 = llvm.getelementptr %36[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c1_i32, %37 : i32, !llvm.ptr
      %38 = llvm.getelementptr %36[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %38 : memref<?xi32>, !llvm.ptr
      %39 = llvm.getelementptr %36[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c110_i32, %39 : i32, !llvm.ptr
      %40 = arith.muli %22, %c4 : index
      %41 = arith.index_cast %40 : index to i64
      %42 = llvm.getelementptr %16[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %43 = llvm.getelementptr %42[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c1_i32, %43 : i32, !llvm.ptr
      %44 = llvm.getelementptr %42[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %44 : memref<?xi32>, !llvm.ptr
      %45 = llvm.getelementptr %42[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c100_i32, %45 : i32, !llvm.ptr
      %46 = arith.muli %22, %c5 : index
      %47 = arith.index_cast %46 : index to i64
      %48 = llvm.getelementptr %16[%47] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %49 = llvm.getelementptr %48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c1_i32, %49 : i32, !llvm.ptr
      %50 = llvm.getelementptr %48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %50 : memref<?xi32>, !llvm.ptr
      %51 = llvm.getelementptr %48[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c105_i32, %51 : i32, !llvm.ptr
      %52 = arith.muli %22, %c6 : index
      %53 = arith.index_cast %52 : index to i64
      %54 = llvm.getelementptr %16[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %55 = llvm.getelementptr %54[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c1_i32, %55 : i32, !llvm.ptr
      %56 = llvm.getelementptr %54[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %56 : memref<?xi32>, !llvm.ptr
      %57 = llvm.getelementptr %54[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c97_i32, %57 : i32, !llvm.ptr
      %58 = arith.muli %22, %c7 : index
      %59 = arith.index_cast %58 : index to i64
      %60 = llvm.getelementptr %16[%59] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %61 = llvm.getelementptr %60[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c1_i32, %61 : i32, !llvm.ptr
      %62 = llvm.getelementptr %60[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %62 : memref<?xi32>, !llvm.ptr
      %63 = llvm.getelementptr %60[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c115_i32, %63 : i32, !llvm.ptr
      %64 = arith.muli %22, %c8 : index
      %65 = arith.index_cast %64 : index to i64
      %66 = llvm.getelementptr %16[%65] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %67 = llvm.getelementptr %66[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c0_i32, %67 : i32, !llvm.ptr
      %68 = llvm.getelementptr %66[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %68 : memref<?xi32>, !llvm.ptr
      %69 = llvm.getelementptr %66[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c112_i32, %69 : i32, !llvm.ptr
      %70 = arith.muli %22, %c9 : index
      %71 = arith.index_cast %70 : index to i64
      %72 = llvm.getelementptr %16[%71] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %73 = llvm.getelementptr %72[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c0_i32, %73 : i32, !llvm.ptr
      %74 = llvm.getelementptr %72[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %74 : memref<?xi32>, !llvm.ptr
      %75 = llvm.getelementptr %72[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c104_i32, %75 : i32, !llvm.ptr
      %76 = arith.muli %22, %c10 : index
      %77 = arith.index_cast %76 : index to i64
      %78 = llvm.getelementptr %16[%77] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %79 = llvm.getelementptr %78[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      %80 = "polygeist.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi8>
      llvm.store %80, %79 : memref<?xi8>, !llvm.ptr
      %81 = llvm.getelementptr %78[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c0_i32, %81 : i32, !llvm.ptr
      %82 = llvm.getelementptr %78[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %20, %82 : memref<?xi32>, !llvm.ptr
      %83 = llvm.getelementptr %78[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>
      llvm.store %c0_i32, %83 : i32, !llvm.ptr
    }
    %10 = llvm.mlir.addressof @str11 : !llvm.ptr
    %cast_6 = memref.cast %2 : memref<11x!llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>> to memref<?x!llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>>
    %cast_7 = memref.cast %alloca_4 : memref<1xi32> to memref<?xi32>
    %11 = "polygeist.pointer2memref"(%10) : (!llvm.ptr) -> memref<?xi8>
    %12:12 = scf.while (%arg2 = %0, %arg3 = %c0_i32, %arg4 = %c1_i32, %arg5 = %cst_1, %arg6 = %c50_i32, %arg7 = %c10_i32, %arg8 = %c10000_i32, %arg9 = %c4_i32, %arg10 = %true, %arg11 = %0, %arg12 = %true, %arg13 = %7) : (i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>) -> (i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, memref<?xi8>, i32) {
      %16 = func.call @getopt_long(%arg0, %arg1, %11, %cast_6, %cast_7) : (i32, memref<?xmemref<?xi8>>, memref<?xi8>, memref<?x!llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>>, memref<?xi32>) -> i32
      %17 = arith.cmpi ne, %16, %c-1_i32 : i32
      %18 = arith.andi %17, %arg12 : i1
      scf.condition(%18) %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg13, %16 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, memref<?xi8>, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32, %arg5: f64, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i1, %arg11: i32, %arg12: memref<?xi8>, %arg13: i32):
      %16:12 = scf.execute_region -> (i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>) {
        cf.switch %arg13 : i32, [
          default: ^bb10(%arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %c1_i32 : i32, i32, i32, f64, i32, i32, i32, i32, i32),
          101: ^bb1,
          109: ^bb2(%arg2 : i32),
          102: ^bb3(%arg2 : i32),
          116: ^bb4(%arg2 : i32),
          110: ^bb5(%arg2, %arg9 : i32, i32),
          100: ^bb6(%arg2, %arg8, %arg9 : i32, i32, i32),
          105: ^bb7(%arg2, %arg7, %arg8, %arg9 : i32, i32, i32, i32),
          97: ^bb8(%arg2, %arg6, %arg7, %arg8, %arg9 : i32, i32, i32, i32, i32),
          115: ^bb9(%arg2, %arg5, %arg6, %arg7, %arg8, %arg9 : i32, f64, i32, i32, i32, i32),
          112: ^bb11(%arg2, %c1_i32, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>),
          104: ^bb10(%arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %c0_i32 : i32, i32, i32, f64, i32, i32, i32, i32, i32)
        ]
      ^bb1:  // pred: ^bb0
        %17 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %18 = affine.load %17[0] : memref<1xmemref<?xi8>>
        %19 = func.call @atoi(%18) : (memref<?xi8>) -> i32
        cf.br ^bb11(%19, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb2(%20: i32):  // pred: ^bb0
        %cast_8 = memref.cast %alloca_5 : memref<16xi8> to memref<?xi8>
        %21 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %22 = affine.load %21[0] : memref<1xmemref<?xi8>>
        %23 = func.call @strncpy(%cast_8, %22, %c15_i64) : (memref<?xi8>, memref<?xi8>, i64) -> memref<?xi8>
        affine.store %c0_i8, %alloca_5[15] : memref<16xi8>
        cf.br ^bb11(%20, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb3(%24: i32):  // pred: ^bb0
        %25 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %26 = affine.load %25[0] : memref<1xmemref<?xi8>>
        cf.br ^bb11(%24, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %true, %26 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb4(%27: i32):  // pred: ^bb0
        %28 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %29 = affine.load %28[0] : memref<1xmemref<?xi8>>
        %30 = func.call @atoi(%29) : (memref<?xi8>) -> i32
        cf.br ^bb11(%27, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %30, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb5(%31: i32, %32: i32):  // pred: ^bb0
        %33 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %34 = affine.load %33[0] : memref<1xmemref<?xi8>>
        %35 = func.call @atoi(%34) : (memref<?xi8>) -> i32
        cf.br ^bb11(%31, %arg3, %arg4, %arg5, %arg6, %arg7, %35, %32, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb6(%36: i32, %37: i32, %38: i32):  // pred: ^bb0
        %39 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %40 = affine.load %39[0] : memref<1xmemref<?xi8>>
        %41 = func.call @atoi(%40) : (memref<?xi8>) -> i32
        cf.br ^bb11(%36, %arg3, %arg4, %arg5, %arg6, %41, %37, %38, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb7(%42: i32, %43: i32, %44: i32, %45: i32):  // pred: ^bb0
        %46 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %47 = affine.load %46[0] : memref<1xmemref<?xi8>>
        %48 = func.call @atoi(%47) : (memref<?xi8>) -> i32
        cf.br ^bb11(%42, %arg3, %arg4, %arg5, %48, %43, %44, %45, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb8(%49: i32, %50: i32, %51: i32, %52: i32, %53: i32):  // pred: ^bb0
        %54 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %55 = affine.load %54[0] : memref<1xmemref<?xi8>>
        %56 = func.call @atof(%55) : (memref<?xi8>) -> f64
        cf.br ^bb11(%49, %arg3, %arg4, %56, %50, %51, %52, %53, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb9(%57: i32, %58: f64, %59: i32, %60: i32, %61: i32, %62: i32):  // pred: ^bb0
        %63 = memref.get_global @optarg : memref<1xmemref<?xi8>>
        %64 = affine.load %63[0] : memref<1xmemref<?xi8>>
        %65 = llvm.mlir.zero : !llvm.ptr
        %66 = "polygeist.pointer2memref"(%65) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
        %67 = func.call @strtoul(%64, %66, %c10_i32) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i64
        %68 = arith.trunci %67 : i64 to i32
        cf.br ^bb11(%57, %arg3, %68, %58, %59, %60, %61, %62, %arg10, %arg11, %true, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb10(%69: i32, %70: i32, %71: i32, %72: f64, %73: i32, %74: i32, %75: i32, %76: i32, %77: i32):  // 2 preds: ^bb0, ^bb0
        %78 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
        func.call @usage(%78) : (memref<?xi8>) -> ()
        cf.br ^bb11(%69, %70, %71, %72, %73, %74, %75, %76, %false, %77, %false, %arg12 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>)
      ^bb11(%79: i32, %80: i32, %81: i32, %82: f64, %83: i32, %84: i32, %85: i32, %86: i32, %87: i1, %88: i32, %89: i1, %90: memref<?xi8>):  // 11 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10
        scf.yield %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>
      }
      scf.yield %16#0, %16#1, %16#2, %16#3, %16#4, %16#5, %16#6, %16#7, %16#8, %16#9, %16#10, %16#11 : i32, i32, i32, f64, i32, i32, i32, i32, i1, i32, i1, memref<?xi8>
    }
    %13 = arith.cmpi ne, %12#1, %c0_i32 : i32
    %14:3 = scf.if %12#8 -> (i32, i1, i32) {
      %16 = arith.cmpi sle, %12#7, %c0_i32 : i32
      %17 = arith.select %16, %c1_i32, %12#7 : i32
      %18 = arith.cmpf ole, %12#3, %cst_0 : f64
      %19 = scf.if %18 -> (i1) {
        scf.yield %true : i1
      } else {
        %21 = arith.cmpf oge, %12#3, %cst : f64
        scf.yield %21 : i1
      }
      %20:2 = scf.if %19 -> (i1, i32) {
        %21 = llvm.mlir.addressof @stderr : !llvm.ptr
        %22 = llvm.load %21 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
        %23 = "polygeist.memref2pointer"(%22) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
        %24 = llvm.mlir.addressof @str12 : !llvm.ptr
        %25 = llvm.getelementptr %24[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
        %26 = llvm.call @fprintf(%23, %25, %12#3) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
        scf.yield %false, %12#9 : i1, i32
      } else {
        %21 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        llvm.store %c0_i32, %22 : i32, !llvm.ptr
        %23 = llvm.getelementptr %21[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        llvm.store %c0_i32, %23 : i32, !llvm.ptr
        %24 = llvm.getelementptr %21[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        %25 = llvm.mlir.zero : !llvm.ptr
        %26 = "polygeist.pointer2memref"(%25) : (!llvm.ptr) -> memref<?xi32>
        llvm.store %26, %24 : memref<?xi32>, !llvm.ptr
        %27 = llvm.getelementptr %21[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        llvm.store %26, %27 : memref<?xi32>, !llvm.ptr
        %28 = llvm.getelementptr %21[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
        llvm.store %26, %28 : memref<?xi32>, !llvm.ptr
        %cast_8 = memref.cast %alloca_5 : memref<16xi8> to memref<?xi8>
        %29 = llvm.mlir.addressof @str2 : !llvm.ptr
        %30 = "polygeist.pointer2memref"(%29) : (!llvm.ptr) -> memref<?xi8>
        %31 = func.call @strcmp(%cast_8, %30) : (memref<?xi8>, memref<?xi8>) -> i32
        %32 = arith.cmpi eq, %31, %c0_i32 : i32
        %33:2 = scf.if %32 -> (i1, i32) {
          %34 = "polygeist.memref2pointer"(%12#10) : (memref<?xi8>) -> !llvm.ptr
          %35 = llvm.mlir.zero : !llvm.ptr
          %36 = llvm.icmp "eq" %34, %35 : !llvm.ptr
          %37 = arith.select %36, %c1_i32, %12#9 : i32
          %38:2 = scf.if %36 -> (i1, i32) {
            %39 = llvm.mlir.addressof @stderr : !llvm.ptr
            %40 = llvm.load %39 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
            %41 = "polygeist.memref2pointer"(%40) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
            %42 = llvm.mlir.addressof @str13 : !llvm.ptr
            %43 = llvm.getelementptr %42[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
            %44 = llvm.call @fprintf(%41, %43) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
            scf.yield %false, %37 : i1, i32
          } else {
            %cast_9 = memref.cast %alloca_3 : memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>> to memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>
            func.call @read_graph_file(%12#10, %cast_9) : (memref<?xi8>, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> ()
            %39 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
            %40 = llvm.load %39 : !llvm.ptr -> i32
            %41 = arith.cmpi eq, %40, %c0_i32 : i32
            %42 = arith.cmpi ne, %40, %c0_i32 : i32
            %43 = arith.select %41, %c1_i32, %37 : i32
            scf.if %41 {
              %44 = llvm.mlir.addressof @stderr : !llvm.ptr
              %45 = llvm.load %44 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
              %46 = "polygeist.memref2pointer"(%45) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
              %47 = llvm.mlir.addressof @str14 : !llvm.ptr
              %48 = llvm.getelementptr %47[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<33 x i8>
              %49 = llvm.call @fprintf(%46, %48) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
            }
            scf.if %42 {
              %44 = llvm.mlir.addressof @stdout : !llvm.ptr
              %45 = llvm.load %44 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
              %46 = "polygeist.memref2pointer"(%45) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
              %47 = llvm.mlir.addressof @str15 : !llvm.ptr
              %48 = llvm.getelementptr %47[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
              %49 = "polygeist.memref2pointer"(%12#10) : (memref<?xi8>) -> !llvm.ptr
              %50 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
              %51 = llvm.load %50 : !llvm.ptr -> i32
              %52 = llvm.getelementptr %50[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
              %53 = llvm.load %52 : !llvm.ptr -> i32
              %54 = llvm.call @fprintf(%46, %48, %49, %51, %53) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> i32
            }
            scf.yield %42, %43 : i1, i32
          }
          scf.yield %38#0, %38#1 : i1, i32
        } else {
          %34 = llvm.mlir.addressof @str0 : !llvm.ptr
          %35 = "polygeist.pointer2memref"(%34) : (!llvm.ptr) -> memref<?xi8>
          %36 = func.call @strcmp(%cast_8, %35) : (memref<?xi8>, memref<?xi8>) -> i32
          %37 = arith.cmpi eq, %36, %c0_i32 : i32
          %38:2 = scf.if %37 -> (i1, i32) {
            %39 = arith.cmpi sle, %12#6, %c0_i32 : i32
            %40 = scf.if %39 -> (i1) {
              scf.yield %true : i1
            } else {
              %43 = arith.cmpi slt, %12#5, %c0_i32 : i32
              scf.yield %43 : i1
            }
            %41 = arith.xori %40, %true : i1
            %42 = arith.select %40, %c1_i32, %12#9 : i32
            scf.if %40 {
              %43 = llvm.mlir.addressof @stderr : !llvm.ptr
              %44 = llvm.load %43 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
              %45 = "polygeist.memref2pointer"(%44) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
              %46 = llvm.mlir.addressof @str16 : !llvm.ptr
              %47 = llvm.getelementptr %46[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<39 x i8>
              %48 = llvm.call @fprintf(%45, %47) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
            } else {
              %cast_9 = memref.cast %alloca_3 : memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>> to memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>
              func.call @make_synthetic_graph(%12#6, %12#5, %12#2, %cast_9) : (i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> ()
              %43 = llvm.mlir.addressof @stdout : !llvm.ptr
              %44 = llvm.load %43 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
              %45 = "polygeist.memref2pointer"(%44) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
              %46 = llvm.mlir.addressof @str17 : !llvm.ptr
              %47 = llvm.getelementptr %46[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<44 x i8>
              %48 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
              %49 = llvm.load %48 : !llvm.ptr -> i32
              %50 = llvm.getelementptr %48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
              %51 = llvm.load %50 : !llvm.ptr -> i32
              %52 = llvm.call @fprintf(%45, %47, %49, %12#5, %51) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> i32
            }
            scf.yield %41, %42 : i1, i32
          } else {
            %39 = llvm.mlir.addressof @stderr : !llvm.ptr
            %40 = llvm.load %39 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
            %41 = "polygeist.memref2pointer"(%40) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
            %42 = llvm.mlir.addressof @str18 : !llvm.ptr
            %43 = llvm.getelementptr %42[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
            %44 = "polygeist.memref2pointer"(%alloca_5) : (memref<16xi8>) -> !llvm.ptr
            %45 = llvm.call @fprintf(%41, %43, %44) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
            scf.yield %false, %c1_i32 : i1, i32
          }
          scf.yield %38#0, %38#1 : i1, i32
        }
        scf.yield %33#0, %33#1 : i1, i32
      }
      scf.yield %17, %20#0, %20#1 : i32, i1, i32
    } else {
      scf.yield %12#7, %false, %12#9 : i32, i1, i32
    }
    %15 = arith.select %14#1, %c0_i32, %14#2 : i32
    scf.if %14#1 {
      %16 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %17 = llvm.load %16 : !llvm.ptr -> i32
      %18 = arith.extsi %17 : i32 to i64
      %19 = arith.muli %18, %c8_i64 : i64
      %20 = func.call @xmalloc(%19) : (i64) -> memref<?xi8>
      %21 = "polygeist.memref2pointer"(%20) : (memref<?xi8>) -> !llvm.ptr
      %22 = "polygeist.pointer2memref"(%21) : (!llvm.ptr) -> memref<?xf64>
      %23 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %24 = llvm.load %23 : !llvm.ptr -> i32
      %25 = arith.extsi %24 : i32 to i64
      %26 = arith.muli %25, %c8_i64 : i64
      %27 = func.call @xmalloc(%26) : (i64) -> memref<?xi8>
      %28 = "polygeist.memref2pointer"(%27) : (memref<?xi8>) -> !llvm.ptr
      %29 = "polygeist.pointer2memref"(%28) : (!llvm.ptr) -> memref<?xf64>
      %30 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %31 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %76 = llvm.load %30 : !llvm.ptr -> i32
        %77 = arith.cmpi slt, %arg2, %76 : i32
        scf.condition(%77) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        %76 = llvm.load %30 : !llvm.ptr -> i32
        %77 = arith.sitofp %76 : i32 to f64
        %78 = arith.divf %cst, %77 : f64
        %79 = llvm.getelementptr %21[%arg2] : (!llvm.ptr, i32) -> !llvm.ptr, f64
        llvm.store %78, %79 : f64, !llvm.ptr
        %80 = llvm.getelementptr %28[%arg2] : (!llvm.ptr, i32) -> !llvm.ptr, f64
        llvm.store %cst_0, %80 : f64, !llvm.ptr
        %81 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %81 : i32
      }
      %cast_8 = memref.cast %alloca_2 : memref<1x!llvm.struct<(i64, array<24 x i8>)>> to memref<?x!llvm.struct<(i64, array<24 x i8>)>>
      %32 = llvm.mlir.zero : !llvm.ptr
      %33 = "polygeist.pointer2memref"(%32) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32)>>
      %34 = func.call @pthread_barrier_init(%cast_8, %33, %14#0) : (memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?x!llvm.struct<(i32)>>, i32) -> i32
      %35 = arith.cmpi ne, %34, %c0_i32 : i32
      scf.if %35 {
        %76 = llvm.mlir.addressof @str19 : !llvm.ptr
        %77 = "polygeist.pointer2memref"(%76) : (!llvm.ptr) -> memref<?xi8>
        func.call @die(%77) : (memref<?xi8>) -> ()
      }
      %36 = arith.extsi %14#0 : i32 to i64
      %37 = arith.muli %36, %c8_i64 : i64
      %38 = func.call @xmalloc(%37) : (i64) -> memref<?xi8>
      %39 = "polygeist.memref2pointer"(%38) : (memref<?xi8>) -> !llvm.ptr
      %40 = "polygeist.pointer2memref"(%39) : (!llvm.ptr) -> memref<?xi64>
      %41 = arith.extsi %14#0 : i32 to i64
      %42 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>}> : () -> index
      %43 = arith.index_cast %42 : index to i64
      %44 = arith.muli %41, %43 : i64
      %45 = func.call @xmalloc(%44) : (i64) -> memref<?xi8>
      %46 = "polygeist.memref2pointer"(%45) : (memref<?xi8>) -> !llvm.ptr
      %47 = "polygeist.pointer2memref"(%46) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>>
      %48 = arith.extsi %14#0 : i32 to i64
      %49 = arith.muli %48, %c8_i64 : i64
      %50 = func.call @xmalloc(%49) : (i64) -> memref<?xi8>
      %51 = "polygeist.memref2pointer"(%50) : (memref<?xi8>) -> !llvm.ptr
      %52 = "polygeist.pointer2memref"(%51) : (!llvm.ptr) -> memref<?xf64>
      affine.store %cst_0, %alloca[0] : memref<1xf64>
      %53 = arith.subf %cst, %12#3 : f64
      %54 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %55 = llvm.load %54 : !llvm.ptr -> i32
      %56 = arith.sitofp %55 : i32 to f64
      %57 = arith.divf %53, %56 : f64
      %58 = arith.index_cast %14#0 : i32 to index
      scf.for %arg2 = %c0 to %58 step %c1 {
        %76 = arith.index_cast %arg2 : index to i32
        %77 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>}> : () -> index
        %78 = arith.muli %arg2, %77 : index
        %79 = arith.index_cast %78 : index to i64
        %80 = llvm.getelementptr %46[%79] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        llvm.store %76, %80 : i32, !llvm.ptr
        %81 = llvm.getelementptr %80[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %14#0, %81 : i32, !llvm.ptr
        %82 = llvm.getelementptr %80[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %83 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %84 = llvm.load %83 : !llvm.ptr -> i32
        llvm.store %84, %82 : i32, !llvm.ptr
        %85 = llvm.getelementptr %80[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %cast_10 = memref.cast %alloca_3 : memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>> to memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>
        llvm.store %cast_10, %85 : memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, !llvm.ptr
        %86 = llvm.getelementptr %80[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %22, %86 : memref<?xf64>, !llvm.ptr
        %87 = llvm.getelementptr %80[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %29, %87 : memref<?xf64>, !llvm.ptr
        %88 = llvm.getelementptr %80[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %12#3, %88 : f64, !llvm.ptr
        %89 = llvm.getelementptr %80[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %57, %89 : f64, !llvm.ptr
        %90 = llvm.getelementptr %80[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %cast_11 = memref.cast %alloca_2 : memref<1x!llvm.struct<(i64, array<24 x i8>)>> to memref<?x!llvm.struct<(i64, array<24 x i8>)>>
        llvm.store %cast_11, %90 : memref<?x!llvm.struct<(i64, array<24 x i8>)>>, !llvm.ptr
        %91 = llvm.getelementptr %80[0, 9] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %52, %91 : memref<?xf64>, !llvm.ptr
        %92 = llvm.getelementptr %80[0, 10] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        %cast_12 = memref.cast %alloca : memref<1xf64> to memref<?xf64>
        llvm.store %cast_12, %92 : memref<?xf64>, !llvm.ptr
        %93 = llvm.getelementptr %80[0, 11] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %12#1, %93 : i32, !llvm.ptr
        %94 = llvm.getelementptr %80[0, 12] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %12#4, %94 : i32, !llvm.ptr
        %95 = llvm.getelementptr %80[0, 13] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>
        llvm.store %12#0, %95 : i32, !llvm.ptr
        %96 = "polygeist.subindex"(%40, %arg2) : (memref<?xi64>, index) -> memref<?xi64>
        %97 = llvm.mlir.zero : !llvm.ptr
        %98 = "polygeist.pointer2memref"(%97) : (!llvm.ptr) -> memref<?x!llvm.struct<(i64, array<48 x i8>)>>
        %99 = "polygeist.get_func"() <{name = @pagerank_worker}> : () -> !llvm.ptr
        %100 = "polygeist.pointer2memref"(%80) : (!llvm.ptr) -> memref<?xi8>
        %101 = "polygeist.pointer2memref"(%99) : (!llvm.ptr) -> memref<?x!llvm.func<memref<?xi8> (memref<?xi8>)>>
        %102 = func.call @pthread_create(%96, %98, %101, %100) : (memref<?xi64>, memref<?x!llvm.struct<(i64, array<48 x i8>)>>, memref<?x!llvm.func<memref<?xi8> (memref<?xi8>)>>, memref<?xi8>) -> i32
        %103 = arith.cmpi ne, %102, %c0_i32 : i32
        scf.if %103 {
          %104 = llvm.mlir.addressof @str20 : !llvm.ptr
          %105 = "polygeist.pointer2memref"(%104) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%105) : (memref<?xi8>) -> ()
        }
      }
      %59 = func.call @now_sec() : () -> f64
      %60 = arith.index_cast %14#0 : i32 to index
      scf.for %arg2 = %c0 to %60 step %c1 {
        %76 = arith.index_cast %arg2 : index to i32
        %77 = llvm.getelementptr %39[%76] : (!llvm.ptr, i32) -> !llvm.ptr, i64
        %78 = llvm.load %77 : !llvm.ptr -> i64
        %79 = llvm.mlir.zero : !llvm.ptr
        %80 = "polygeist.pointer2memref"(%79) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
        %81 = func.call @pthread_join(%78, %80) : (i64, memref<?xmemref<?xi8>>) -> i32
      }
      %61 = func.call @now_sec() : () -> f64
      %62 = llvm.mlir.addressof @str21 : !llvm.ptr
      %63 = llvm.getelementptr %62[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %64 = arith.subf %61, %59 : f64
      %65 = llvm.call @printf(%63, %64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      scf.if %13 {
        %76 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %77 = llvm.load %76 : !llvm.ptr -> i32
        %78 = arith.extsi %77 : i32 to i64
        %79 = arith.muli %78, %c16_i64 : i64
        %80 = func.call @xmalloc(%79) : (i64) -> memref<?xi8>
        %81 = "polygeist.memref2pointer"(%80) : (memref<?xi8>) -> !llvm.ptr
        %82 = "polygeist.pointer2memref"(%81) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
        %83 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %84:4 = scf.while (%arg2 = %c0_i32, %arg3 = %cst_0, %arg4 = %cst, %arg5 = %cst_0) : (i32, f64, f64, f64) -> (f64, f64, f64, i32) {
          %132 = llvm.load %83 : !llvm.ptr -> i32
          %133 = arith.cmpi slt, %arg2, %132 : i32
          scf.condition(%133) %arg3, %arg4, %arg5, %arg2 : f64, f64, f64, i32
        } do {
        ^bb0(%arg2: f64, %arg3: f64, %arg4: f64, %arg5: i32):
          %132 = arith.index_cast %arg5 : i32 to index
          %133 = arith.muli %132, %c16 : index
          %134 = arith.index_cast %133 : index to i64
          %135 = llvm.getelementptr %81[%134] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          llvm.store %arg5, %135 : i32, !llvm.ptr
          %136 = llvm.getelementptr %135[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
          %137 = llvm.getelementptr %21[%arg5] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %138 = llvm.load %137 : !llvm.ptr -> f64
          llvm.store %138, %136 : f64, !llvm.ptr
          %139 = llvm.getelementptr %21[%arg5] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %140 = llvm.load %139 : !llvm.ptr -> f64
          %141 = arith.addf %arg4, %140 : f64
          %142 = arith.cmpf olt, %140, %arg3 : f64
          %143 = scf.if %142 -> (f64) {
            %147 = llvm.getelementptr %21[%arg5] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            %148 = llvm.load %147 : !llvm.ptr -> f64
            scf.yield %148 : f64
          } else {
            scf.yield %arg3 : f64
          }
          %144 = arith.cmpf ogt, %140, %arg2 : f64
          %145 = scf.if %144 -> (f64) {
            %147 = llvm.getelementptr %21[%arg5] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            %148 = llvm.load %147 : !llvm.ptr -> f64
            scf.yield %148 : f64
          } else {
            scf.yield %arg2 : f64
          }
          %146 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %146, %145, %143, %141 : i32, f64, f64, f64
        }
        %85 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %86 = llvm.load %85 : !llvm.ptr -> i32
        %87 = arith.extsi %86 : i32 to i64
        %88 = "polygeist.get_func"() <{name = @compare_pagerank}> : () -> !llvm.ptr
        %89 = "polygeist.pointer2memref"(%88) : (!llvm.ptr) -> memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>
        func.call @qsort(%80, %87, %c16_i64, %89) : (memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) -> ()
        %90 = llvm.mlir.addressof @str22 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<30 x i8>
        %92 = llvm.call @printf(%91) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        %93 = llvm.mlir.addressof @str23 : !llvm.ptr
        %94 = llvm.getelementptr %93[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
        %95 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %96 = llvm.load %95 : !llvm.ptr -> i32
        %97 = llvm.call @printf(%94, %96) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        %98 = llvm.mlir.addressof @str24 : !llvm.ptr
        %99 = llvm.getelementptr %98[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<40 x i8>
        %100 = llvm.call @printf(%99, %84#2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
        %101 = llvm.mlir.addressof @str25 : !llvm.ptr
        %102 = llvm.getelementptr %101[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
        %103 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %104 = llvm.load %103 : !llvm.ptr -> i32
        %105 = arith.sitofp %104 : i32 to f64
        %106 = arith.divf %84#2, %105 : f64
        %107 = llvm.call @printf(%102, %106) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
        %108 = llvm.mlir.addressof @str26 : !llvm.ptr
        %109 = llvm.getelementptr %108[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
        %110 = llvm.call @printf(%109, %84#1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
        %111 = llvm.mlir.addressof @str27 : !llvm.ptr
        %112 = llvm.getelementptr %111[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
        %113 = llvm.call @printf(%112, %84#0) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
        %114 = llvm.mlir.addressof @str28 : !llvm.ptr
        %115 = llvm.getelementptr %114[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<30 x i8>
        %116 = llvm.call @printf(%115) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        %117 = llvm.mlir.addressof @str29 : !llvm.ptr
        %118 = llvm.getelementptr %117[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<37 x i8>
        %119 = llvm.call @printf(%118) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        %120 = llvm.mlir.addressof @str30 : !llvm.ptr
        %121 = llvm.getelementptr %120[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
        %122 = llvm.call @printf(%121) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        %123 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %124 = llvm.load %123 : !llvm.ptr -> i32
        %125 = arith.cmpi slt, %124, %c20_i32 : i32
        %126 = scf.if %125 -> (i32) {
          %132 = llvm.load %123 : !llvm.ptr -> i32
          scf.yield %132 : i32
        } else {
          scf.yield %c20_i32 : i32
        }
        %127 = arith.index_cast %126 : i32 to index
        scf.for %arg2 = %c0 to %127 step %c1 {
          %132 = arith.index_cast %arg2 : index to i32
          %133 = llvm.mlir.addressof @str31 : !llvm.ptr
          %134 = llvm.getelementptr %133[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
          %135 = arith.addi %132, %c1_i32 : i32
          %136 = arith.muli %arg2, %c16 : index
          %137 = arith.index_cast %136 : index to i64
          %138 = llvm.getelementptr %81[%137] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %139 = llvm.load %138 : !llvm.ptr -> i32
          %140 = llvm.getelementptr %138[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
          %141 = llvm.load %140 : !llvm.ptr -> f64
          %142 = llvm.call @printf(%134, %135, %139, %141) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, f64) -> i32
        }
        %128 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %129 = llvm.mlir.addressof @str32 : !llvm.ptr
        %130 = llvm.getelementptr %129[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i8>
        %131 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
          %132 = llvm.load %128 : !llvm.ptr -> i32
          %133 = arith.cmpi slt, %arg2, %132 : i32
          scf.condition(%133) %arg2 : i32
        } do {
        ^bb0(%arg2: i32):
          %132 = llvm.getelementptr %21[%arg2] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %133 = llvm.load %132 : !llvm.ptr -> f64
          %134 = llvm.call @printf(%130, %arg2, %133) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, f64) -> i32
          %135 = arith.addi %arg2, %c1_i32 : i32
          scf.yield %135 : i32
        }
        memref.dealloc %82 : memref<?x!llvm.struct<(i32, f64)>>
      }
      %cast_9 = memref.cast %alloca_2 : memref<1x!llvm.struct<(i64, array<24 x i8>)>> to memref<?x!llvm.struct<(i64, array<24 x i8>)>>
      %66 = func.call @pthread_barrier_destroy(%cast_9) : (memref<?x!llvm.struct<(i64, array<24 x i8>)>>) -> i32
      memref.dealloc %40 : memref<?xi64>
      memref.dealloc %47 : memref<?x!llvm.struct<(i32, i32, i32, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>, memref<?xf64>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?xf64>, memref<?xf64>, i32, i32, i32)>>
      memref.dealloc %52 : memref<?xf64>
      memref.dealloc %22 : memref<?xf64>
      memref.dealloc %29 : memref<?xf64>
      %67 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %68 = llvm.getelementptr %67[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
      %69 = llvm.load %68 : !llvm.ptr -> memref<?xi32>
      memref.dealloc %69 : memref<?xi32>
      %70 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %71 = llvm.getelementptr %70[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
      %72 = llvm.load %71 : !llvm.ptr -> memref<?xi32>
      memref.dealloc %72 : memref<?xi32>
      %73 = "polygeist.memref2pointer"(%alloca_3) : (memref<1x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
      %74 = llvm.getelementptr %73[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
      %75 = llvm.load %74 : !llvm.ptr -> memref<?xi32>
      memref.dealloc %75 : memref<?xi32>
    }
    return %15 : i32
  }
  func.func private @strncpy(memref<?xi8>, memref<?xi8>, i64) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @getopt_long(i32, memref<?xmemref<?xi8>>, memref<?xi8>, memref<?x!llvm.struct<(memref<?xi8>, i32, memref<?xi32>, i32)>>, memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atof(memref<?xi8>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strtoul(memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @usage(%arg0: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %2 = "polygeist.memref2pointer"(%1) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %3 = llvm.mlir.addressof @str33 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<617 x i8>
    %5 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %6 = llvm.call @fprintf(%2, %4, %5) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    return
  }
  func.func private @strcmp(memref<?xi8>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @read_graph_file(%arg0: memref<?xi8>, %arg1: memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<1xi32>
    %0 = llvm.mlir.undef : i32
    affine.store %0, %alloca[0] : memref<1xi32>
    %alloca_0 = memref.alloca() : memref<1xi32>
    affine.store %0, %alloca_0[0] : memref<1xi32>
    %1 = llvm.mlir.undef : i64
    %alloca_1 = memref.alloca() : memref<1xi64>
    affine.store %1, %alloca_1[0] : memref<1xi64>
    %alloca_2 = memref.alloca() : memref<1xmemref<?xi8>>
    %alloca_3 = memref.alloca() : memref<1x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>
    %2 = llvm.mlir.addressof @str34 : !llvm.ptr
    %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi8>
    %4 = call @fopen(%arg0, %3) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %5 = "polygeist.memref2pointer"(%4) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.icmp "eq" %5, %6 : !llvm.ptr
    scf.if %7 {
      %11 = llvm.mlir.addressof @stderr : !llvm.ptr
      %12 = llvm.load %11 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %13 = "polygeist.memref2pointer"(%12) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %14 = llvm.mlir.addressof @str35 : !llvm.ptr
      %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x i8>
      %16 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %17 = func.call @__errno_location() : () -> memref<?xi32>
      %18 = affine.load %17[0] : memref<?xi32>
      %19 = func.call @strerror(%18) : (i32) -> memref<?xi8>
      %20 = "polygeist.memref2pointer"(%19) : (memref<?xi8>) -> !llvm.ptr
      %21 = llvm.call @fprintf(%13, %15, %16, %20) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %cast = memref.cast %alloca_3 : memref<1x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>> to memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>
    call @el_init(%cast) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) -> ()
    %8 = "polygeist.pointer2memref"(%6) : (!llvm.ptr) -> memref<?xi8>
    affine.store %8, %alloca_2[0] : memref<1xmemref<?xi8>>
    affine.store %c0_i64, %alloca_1[0] : memref<1xi64>
    %cast_4 = memref.cast %alloca_2 : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
    %cast_5 = memref.cast %alloca_1 : memref<1xi64> to memref<?xi64>
    scf.while : () -> () {
      %11 = func.call @getline(%cast_4, %cast_5, %4) : (memref<?xmemref<?xi8>>, memref<?xi64>, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
      %12 = arith.cmpi ne, %11, %c-1_i64 : i64
      %13 = arith.cmpi ne, %11, %c0_i64 : i64
      %14 = arith.andi %12, %13 : i1
      scf.if %14 {
        %15 = affine.load %alloca_2[0] : memref<1xmemref<?xi8>>
        %16 = "polygeist.memref2pointer"(%15) : (memref<?xi8>) -> !llvm.ptr
        %17 = llvm.mlir.addressof @str36 : !llvm.ptr
        %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
        %19 = "polygeist.memref2pointer"(%alloca_0) : (memref<1xi32>) -> !llvm.ptr
        %20 = "polygeist.memref2pointer"(%alloca) : (memref<1xi32>) -> !llvm.ptr
        %21 = llvm.call @__isoc99_sscanf(%16, %18, %19, %20) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
        %22 = arith.cmpi eq, %21, %c2_i32 : i32
        scf.if %22 {
          %23 = affine.load %alloca_0[0] : memref<1xi32>
          %24 = arith.cmpi slt, %23, %c0_i32 : i32
          %25 = scf.if %24 -> (i1) {
            scf.yield %true : i1
          } else {
            %27 = affine.load %alloca[0] : memref<1xi32>
            %28 = arith.cmpi slt, %27, %c0_i32 : i32
            scf.yield %28 : i1
          }
          %26 = arith.xori %25, %true : i1
          scf.if %26 {
            %27 = affine.load %alloca[0] : memref<1xi32>
            func.call @el_push(%cast, %23, %27) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>, i32, i32) -> ()
          }
        }
      }
      scf.condition(%12)
    } do {
      scf.yield
    }
    %9 = affine.load %alloca_2[0] : memref<1xmemref<?xi8>>
    memref.dealloc %9 : memref<?xi8>
    %10 = call @fclose(%4) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
    call @build_csr_from_edges(%cast, %arg1) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> ()
    call @el_free(%cast) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) -> ()
    return
  }
  func.func private @make_synthetic_graph(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<1x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>
    %cast = memref.cast %alloca : memref<1x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>> to memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>
    call @el_init(%cast) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) -> ()
    call @srand(%arg2) : (i32) -> ()
    %0 = arith.index_cast %arg0 : i32 to index
    scf.for %arg4 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg4 : index to i32
      %2 = arith.index_cast %arg1 : i32 to index
      scf.for %arg5 = %c0 to %2 step %c1 {
        %3 = func.call @rand() : () -> i32
        %4 = arith.remsi %3, %arg0 : i32
        %5 = arith.cmpi eq, %4, %1 : i32
        %6 = scf.if %5 -> (i32) {
          %7 = arith.addi %4, %c1_i32 : i32
          %8 = arith.remsi %7, %arg0 : i32
          scf.yield %8 : i32
        } else {
          scf.yield %4 : i32
        }
        func.call @el_push(%cast, %6, %1) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>, i32, i32) -> ()
      }
    }
    call @build_csr_from_edges(%cast, %arg3) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>, memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> ()
    call @el_free(%cast) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) -> ()
    return
  }
  func.func private @xmalloc(%arg0: i64) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = call @malloc(%arg0) : (i64) -> memref<?xi8>
    %1 = "polygeist.memref2pointer"(%0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    scf.if %3 {
      %4 = llvm.mlir.addressof @str37 : !llvm.ptr
      %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%5) : (memref<?xi8>) -> ()
    }
    return %0 : memref<?xi8>
  }
  func.func private @pthread_barrier_init(memref<?x!llvm.struct<(i64, array<24 x i8>)>>, memref<?x!llvm.struct<(i32)>>, i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @die(%arg0: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1_i32 = arith.constant 1 : i32
    call @perror(%arg0) : (memref<?xi8>) -> ()
    call @exit(%c1_i32) : (i32) -> ()
    return
  }
  func.func private @pthread_create(memref<?xi64>, memref<?x!llvm.struct<(i64, array<48 x i8>)>>, memref<?x!llvm.func<memref<?xi8> (memref<?xi8>)>>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @now_sec() -> f64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 1.000000e+09 : f64
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<1x2xi64>
    %cast = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
    %0 = call @clock_gettime(%c1_i32, %cast) : (i32, memref<?x2xi64>) -> i32
    %1 = affine.load %alloca[0, 0] : memref<1x2xi64>
    %2 = arith.sitofp %1 : i64 to f64
    %3 = affine.load %alloca[0, 1] : memref<1x2xi64>
    %4 = arith.sitofp %3 : i64 to f64
    %5 = arith.divf %4, %cst : f64
    %6 = arith.addf %2, %5 : f64
    return %6 : f64
  }
  func.func private @pthread_join(i64, memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @qsort(memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @compare_pagerank(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
    %3 = llvm.load %2 : !llvm.ptr -> f64
    %4 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
    %5 = llvm.getelementptr %4[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
    %6 = llvm.load %5 : !llvm.ptr -> f64
    %7 = arith.cmpf olt, %3, %6 : f64
    %8 = arith.select %7, %c1_i32, %0 : i32
    %9:2 = scf.if %7 -> (i1, i32) {
      scf.yield %false, %8 : i1, i32
    } else {
      %11 = arith.cmpf ogt, %3, %6 : f64
      %12 = arith.xori %11, %true : i1
      %13 = arith.select %11, %c-1_i32, %8 : i32
      scf.yield %12, %13 : i1, i32
    }
    %10 = arith.select %9#0, %c0_i32, %9#1 : i32
    return %10 : i32
  }
  func.func private @pthread_barrier_destroy(memref<?x!llvm.struct<(i64, array<24 x i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fopen(memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strerror(i32) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__errno_location() -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @el_init(%arg0: memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) -> !llvm.ptr
    %1 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi32>
    llvm.store %3, %1 : memref<?xi32>, !llvm.ptr
    %4 = llvm.load %1 : !llvm.ptr -> memref<?xi32>
    llvm.store %4, %0 : memref<?xi32>, !llvm.ptr
    %5 = llvm.getelementptr %0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    llvm.store %c0_i32, %5 : i32, !llvm.ptr
    %6 = llvm.getelementptr %0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    llvm.store %c0_i32, %6 : i32, !llvm.ptr
    return
  }
  func.func private @getline(memref<?xmemref<?xi8>>, memref<?xi64>, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @el_push(%arg0: memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>, %arg1: i32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %true = arith.constant true
    %c4_i64 = arith.constant 4 : i64
    %c1_i32 = arith.constant 1 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) -> !llvm.ptr
    %1 = llvm.getelementptr %0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %2 = llvm.load %1 : !llvm.ptr -> i32
    %3 = llvm.getelementptr %0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %4 = llvm.load %3 : !llvm.ptr -> i32
    %5 = arith.cmpi eq, %2, %4 : i32
    scf.if %5 {
      %15 = llvm.load %3 : !llvm.ptr -> i32
      %16 = arith.cmpi ne, %15, %c0_i32 : i32
      %17 = scf.if %16 -> (i32) {
        %38 = llvm.load %3 : !llvm.ptr -> i32
        %39 = arith.muli %38, %c2_i32 : i32
        scf.yield %39 : i32
      } else {
        scf.yield %c4096_i32 : i32
      }
      %18 = llvm.load %0 : !llvm.ptr -> memref<?xi32>
      %19 = "polygeist.memref2pointer"(%18) : (memref<?xi32>) -> !llvm.ptr
      %20 = "polygeist.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi8>
      %21 = arith.extsi %17 : i32 to i64
      %22 = arith.muli %21, %c4_i64 : i64
      %23 = func.call @realloc(%20, %22) : (memref<?xi8>, i64) -> memref<?xi8>
      %24 = "polygeist.memref2pointer"(%23) : (memref<?xi8>) -> !llvm.ptr
      %25 = "polygeist.pointer2memref"(%24) : (!llvm.ptr) -> memref<?xi32>
      llvm.store %25, %0 : memref<?xi32>, !llvm.ptr
      %26 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
      %27 = llvm.load %26 : !llvm.ptr -> memref<?xi32>
      %28 = "polygeist.memref2pointer"(%27) : (memref<?xi32>) -> !llvm.ptr
      %29 = "polygeist.pointer2memref"(%28) : (!llvm.ptr) -> memref<?xi8>
      %30 = func.call @realloc(%29, %22) : (memref<?xi8>, i64) -> memref<?xi8>
      %31 = "polygeist.memref2pointer"(%30) : (memref<?xi8>) -> !llvm.ptr
      %32 = "polygeist.pointer2memref"(%31) : (!llvm.ptr) -> memref<?xi32>
      llvm.store %32, %26 : memref<?xi32>, !llvm.ptr
      %33 = llvm.load %0 : !llvm.ptr -> memref<?xi32>
      %34 = "polygeist.memref2pointer"(%33) : (memref<?xi32>) -> !llvm.ptr
      %35 = llvm.mlir.zero : !llvm.ptr
      %36 = llvm.icmp "eq" %34, %35 : !llvm.ptr
      %37 = scf.if %36 -> (i1) {
        scf.yield %true : i1
      } else {
        %38 = llvm.load %26 : !llvm.ptr -> memref<?xi32>
        %39 = "polygeist.memref2pointer"(%38) : (memref<?xi32>) -> !llvm.ptr
        %40 = llvm.icmp "eq" %39, %35 : !llvm.ptr
        scf.yield %40 : i1
      }
      scf.if %37 {
        %38 = llvm.mlir.addressof @str38 : !llvm.ptr
        %39 = "polygeist.pointer2memref"(%38) : (!llvm.ptr) -> memref<?xi8>
        func.call @die(%39) : (memref<?xi8>) -> ()
      }
      llvm.store %17, %3 : i32, !llvm.ptr
    }
    %6 = llvm.load %0 : !llvm.ptr -> memref<?xi32>
    %7 = llvm.load %1 : !llvm.ptr -> i32
    %8 = arith.index_cast %7 : i32 to index
    affine.store %arg1, %6[symbol(%8)] : memref<?xi32>
    %9 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %10 = llvm.load %9 : !llvm.ptr -> memref<?xi32>
    %11 = llvm.load %1 : !llvm.ptr -> i32
    %12 = arith.index_cast %11 : i32 to index
    affine.store %arg2, %10[symbol(%12)] : memref<?xi32>
    %13 = llvm.load %1 : !llvm.ptr -> i32
    %14 = arith.addi %13, %c1_i32 : i32
    llvm.store %14, %1 : i32, !llvm.ptr
    return
  }
  func.func private @fclose(memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @build_csr_from_edges(%arg0: memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>, %arg1: memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) -> !llvm.ptr
    %2 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %3 = llvm.load %2 : !llvm.ptr -> i32
    %4 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %5 = arith.index_cast %3 : i32 to index
    %6 = scf.for %arg2 = %c0 to %5 step %c1 iter_args(%arg3 = %c-1_i32) -> (i32) {
      %76 = llvm.load %1 : !llvm.ptr -> memref<?xi32>
      %77 = memref.load %76[%arg2] : memref<?xi32>
      %78 = arith.cmpi sgt, %77, %arg3 : i32
      %79 = scf.if %78 -> (i32) {
        %84 = llvm.load %1 : !llvm.ptr -> memref<?xi32>
        %85 = memref.load %84[%arg2] : memref<?xi32>
        scf.yield %85 : i32
      } else {
        scf.yield %arg3 : i32
      }
      %80 = llvm.load %4 : !llvm.ptr -> memref<?xi32>
      %81 = memref.load %80[%arg2] : memref<?xi32>
      %82 = arith.cmpi sgt, %81, %79 : i32
      %83 = scf.if %82 -> (i32) {
        %84 = llvm.load %4 : !llvm.ptr -> memref<?xi32>
        %85 = memref.load %84[%arg2] : memref<?xi32>
        scf.yield %85 : i32
      } else {
        scf.yield %79 : i32
      }
      scf.yield %83 : i32
    }
    %7 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %8 = arith.cmpi sge, %6, %c0_i32 : i32
    %9 = scf.if %8 -> (i32) {
      %76 = arith.addi %6, %c1_i32 : i32
      scf.yield %76 : i32
    } else {
      scf.yield %c0_i32 : i32
    }
    llvm.store %9, %7 : i32, !llvm.ptr
    %10 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %11 = llvm.getelementptr %10[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %12 = llvm.load %2 : !llvm.ptr -> i32
    llvm.store %12, %11 : i32, !llvm.ptr
    %13 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %14 = llvm.getelementptr %13[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %15 = llvm.load %13 : !llvm.ptr -> i32
    %16 = arith.addi %15, %c1_i32 : i32
    %17 = arith.extsi %16 : i32 to i64
    %18 = arith.muli %17, %c4_i64 : i64
    %19 = call @xmalloc(%18) : (i64) -> memref<?xi8>
    %20 = "polygeist.memref2pointer"(%19) : (memref<?xi8>) -> !llvm.ptr
    %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi32>
    llvm.store %21, %14 : memref<?xi32>, !llvm.ptr
    %22 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %23 = llvm.getelementptr %22[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %24 = llvm.getelementptr %22[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %25 = llvm.load %24 : !llvm.ptr -> i32
    %26 = arith.extsi %25 : i32 to i64
    %27 = arith.muli %26, %c4_i64 : i64
    %28 = call @xmalloc(%27) : (i64) -> memref<?xi8>
    %29 = "polygeist.memref2pointer"(%28) : (memref<?xi8>) -> !llvm.ptr
    %30 = "polygeist.pointer2memref"(%29) : (!llvm.ptr) -> memref<?xi32>
    llvm.store %30, %23 : memref<?xi32>, !llvm.ptr
    %31 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %32 = llvm.getelementptr %31[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %33 = llvm.load %31 : !llvm.ptr -> i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = arith.muli %34, %c4 : index
    %36 = arith.divui %35, %c4 : index
    %alloc = memref.alloc(%36) : memref<?xi32>
    scf.for %arg2 = %c0 to %36 step %c1 {
      memref.store %c0_i32, %alloc[%arg2] : memref<?xi32>
    }
    llvm.store %alloc, %32 : memref<?xi32>, !llvm.ptr
    %37 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %38 = llvm.getelementptr %37[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %39 = llvm.load %38 : !llvm.ptr -> memref<?xi32>
    %40 = "polygeist.memref2pointer"(%39) : (memref<?xi32>) -> !llvm.ptr
    %41 = llvm.mlir.zero : !llvm.ptr
    %42 = llvm.icmp "eq" %40, %41 : !llvm.ptr
    scf.if %42 {
      %76 = llvm.mlir.addressof @str39 : !llvm.ptr
      %77 = "polygeist.pointer2memref"(%76) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%77) : (memref<?xi8>) -> ()
    }
    %43 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %44 = llvm.load %43 : !llvm.ptr -> i32
    %45 = arith.index_cast %44 : i32 to index
    %46 = arith.muli %45, %c4 : index
    %47 = arith.divui %46, %c4 : index
    %alloc_0 = memref.alloc(%47) : memref<?xi32>
    scf.for %arg2 = %c0 to %47 step %c1 {
      memref.store %c0_i32, %alloc_0[%arg2] : memref<?xi32>
    }
    %48 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xi32>) -> !llvm.ptr
    %49 = llvm.mlir.zero : !llvm.ptr
    %50 = llvm.icmp "eq" %48, %49 : !llvm.ptr
    scf.if %50 {
      %76 = llvm.mlir.addressof @str39 : !llvm.ptr
      %77 = "polygeist.pointer2memref"(%76) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%77) : (memref<?xi8>) -> ()
    }
    %51 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %52 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %76 = llvm.load %2 : !llvm.ptr -> i32
      %77 = arith.cmpi slt, %arg2, %76 : i32
      scf.condition(%77) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):
      %76 = llvm.load %1 : !llvm.ptr -> memref<?xi32>
      %77 = arith.index_cast %arg2 : i32 to index
      %78 = memref.load %76[%77] : memref<?xi32>
      %79 = llvm.load %51 : !llvm.ptr -> memref<?xi32>
      %80 = memref.load %79[%77] : memref<?xi32>
      %81 = arith.cmpi slt, %78, %c0_i32 : i32
      %82 = scf.if %81 -> (i1) {
        scf.yield %true : i1
      } else {
        %85 = arith.cmpi slt, %80, %c0_i32 : i32
        scf.yield %85 : i1
      }
      %83 = arith.xori %82, %true : i1
      scf.if %83 {
        %85 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %86 = llvm.load %85 : !llvm.ptr -> i32
        %87 = arith.cmpi sge, %78, %86 : i32
        %88 = scf.if %87 -> (i1) {
          scf.yield %true : i1
        } else {
          %90 = llvm.load %85 : !llvm.ptr -> i32
          %91 = arith.cmpi sge, %80, %90 : i32
          scf.yield %91 : i1
        }
        %89 = arith.xori %88, %true : i1
        scf.if %89 {
          %90 = arith.index_cast %80 : i32 to index
          %91 = memref.load %alloc_0[%90] : memref<?xi32>
          %92 = arith.addi %91, %c1_i32 : i32
          memref.store %92, %alloc_0[%90] : memref<?xi32>
          %93 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
          %94 = llvm.getelementptr %93[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
          %95 = llvm.load %94 : !llvm.ptr -> memref<?xi32>
          %96 = arith.index_cast %78 : i32 to index
          %97 = memref.load %95[%96] : memref<?xi32>
          %98 = arith.addi %97, %c1_i32 : i32
          memref.store %98, %95[%96] : memref<?xi32>
        }
      }
      %84 = arith.addi %arg2, %c1_i32 : i32
      scf.yield %84 : i32
    }
    %53 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %54 = llvm.getelementptr %53[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %55 = llvm.load %54 : !llvm.ptr -> memref<?xi32>
    affine.store %c0_i32, %55[0] : memref<?xi32>
    %56 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %57 = llvm.getelementptr %56[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %58 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %76 = llvm.load %56 : !llvm.ptr -> i32
      %77 = arith.cmpi slt, %arg2, %76 : i32
      scf.condition(%77) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):
      %76 = llvm.load %57 : !llvm.ptr -> memref<?xi32>
      %77 = arith.addi %arg2, %c1_i32 : i32
      %78 = arith.index_cast %77 : i32 to index
      %79 = arith.index_cast %arg2 : i32 to index
      %80 = memref.load %76[%79] : memref<?xi32>
      %81 = memref.load %alloc_0[%79] : memref<?xi32>
      %82 = arith.addi %80, %81 : i32
      memref.store %82, %76[%78] : memref<?xi32>
      %83 = arith.addi %arg2, %c1_i32 : i32
      scf.yield %83 : i32
    }
    %59 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %60 = llvm.load %59 : !llvm.ptr -> i32
    %61 = arith.extsi %60 : i32 to i64
    %62 = arith.muli %61, %c4_i64 : i64
    %63 = call @xmalloc(%62) : (i64) -> memref<?xi8>
    %64 = "polygeist.memref2pointer"(%63) : (memref<?xi8>) -> !llvm.ptr
    %65 = "polygeist.pointer2memref"(%64) : (!llvm.ptr) -> memref<?xi32>
    %66 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
    %67 = llvm.getelementptr %66[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
    %68 = llvm.load %67 : !llvm.ptr -> memref<?xi32>
    %69 = "polygeist.memref2pointer"(%68) : (memref<?xi32>) -> !llvm.ptr
    %70 = llvm.load %66 : !llvm.ptr -> i32
    %71 = arith.extsi %70 : i32 to i64
    %72 = arith.muli %71, %c4_i64 : i64
    %73 = arith.index_cast %72 : i64 to index
    scf.for %arg2 = %c0 to %73 step %c1 {
      %76 = arith.index_cast %arg2 : index to i32
      %77 = llvm.getelementptr %69[%76] : (!llvm.ptr, i32) -> !llvm.ptr, i8
      %78 = llvm.load %77 : !llvm.ptr -> i8
      %79 = llvm.getelementptr %64[%76] : (!llvm.ptr, i32) -> !llvm.ptr, i8
      llvm.store %78, %79 : i8, !llvm.ptr
    }
    %74 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %75:2 = scf.while (%arg2 = %0, %arg3 = %c0_i32) : (i32, i32) -> (i32, i32) {
      %76 = llvm.load %2 : !llvm.ptr -> i32
      %77 = arith.cmpi slt, %arg3, %76 : i32
      scf.condition(%77) %arg2, %arg3 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):
      %76 = llvm.load %1 : !llvm.ptr -> memref<?xi32>
      %77 = arith.index_cast %arg3 : i32 to index
      %78 = memref.load %76[%77] : memref<?xi32>
      %79 = llvm.load %74 : !llvm.ptr -> memref<?xi32>
      %80 = memref.load %79[%77] : memref<?xi32>
      %81 = arith.cmpi slt, %78, %c0_i32 : i32
      %82 = scf.if %81 -> (i1) {
        scf.yield %true : i1
      } else {
        %85 = arith.cmpi slt, %80, %c0_i32 : i32
        scf.yield %85 : i1
      }
      %83 = scf.if %82 -> (i32) {
        scf.yield %arg2 : i32
      } else {
        %85 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
        %86 = llvm.load %85 : !llvm.ptr -> i32
        %87 = arith.cmpi sge, %78, %86 : i32
        %88 = scf.if %87 -> (i1) {
          scf.yield %true : i1
        } else {
          %90 = llvm.load %85 : !llvm.ptr -> i32
          %91 = arith.cmpi sge, %80, %90 : i32
          scf.yield %91 : i1
        }
        %89 = scf.if %88 -> (i32) {
          scf.yield %arg2 : i32
        } else {
          %90 = llvm.getelementptr %64[%80] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          %91 = llvm.load %90 : !llvm.ptr -> i32
          %92 = arith.addi %91, %c1_i32 : i32
          %93 = llvm.getelementptr %64[%80] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          llvm.store %92, %93 : i32, !llvm.ptr
          %94 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>>) -> !llvm.ptr
          %95 = llvm.getelementptr %94[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi32>, memref<?xi32>, memref<?xi32>)>
          %96 = llvm.load %95 : !llvm.ptr -> memref<?xi32>
          %97 = arith.index_cast %91 : i32 to index
          memref.store %78, %96[%97] : memref<?xi32>
          scf.yield %91 : i32
        }
        scf.yield %89 : i32
      }
      %84 = arith.addi %arg3, %c1_i32 : i32
      scf.yield %83, %84 : i32, i32
    }
    memref.dealloc %65 : memref<?xi32>
    memref.dealloc %alloc_0 : memref<?xi32>
    return
  }
  func.func private @el_free(%arg0: memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>>) -> !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> memref<?xi32>
    memref.dealloc %1 : memref<?xi32>
    %2 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %3 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
    memref.dealloc %3 : memref<?xi32>
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr) -> memref<?xi32>
    llvm.store %5, %2 : memref<?xi32>, !llvm.ptr
    %6 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
    llvm.store %6, %0 : memref<?xi32>, !llvm.ptr
    %7 = llvm.getelementptr %0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    %8 = llvm.getelementptr %0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>, memref<?xi32>, i32, i32)>
    llvm.store %c0_i32, %8 : i32, !llvm.ptr
    %9 = llvm.load %8 : !llvm.ptr -> i32
    llvm.store %9, %7 : i32, !llvm.ptr
    return
  }
  func.func private @srand(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @malloc(i64) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @perror(memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @clock_gettime(i32, memref<?x2xi64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @realloc(memref<?xi8>, i64) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
}
