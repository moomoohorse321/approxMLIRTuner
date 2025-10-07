module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  // === Knob 1: Per-term scoring loop over documents (loop_perforate) ========
  // Exact func name: score_term_over_docs
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "score_term_over_docs",
    transform_type = "loop_perforate",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 20>,
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 0, 1, 2, 3>,
    thresholds = array<i32: 19>,      // e.g., perforate when many docs
    decisions = array<i32: 1, 0>
  }> : () -> ()


  // === Knob 2: rank_documents_bm25 (func_substitute) ================================
  // Exact func name: rank_documents_bm25
  // Approx func name present in C: approx_rank_documents_bm25
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "rank_documents_bm25",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 5>,
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 0, 1, 2>,
    thresholds = array<i32: 2>,
    decisions = array<i32: 0, 1>
  }> : () -> ()
  // Required for func_substitute:
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "rank_documents_bm25"}> : () -> ()

  // === Knob 3: count_and_lower_words (func_substitute) =========================
  // Exact func name: count_and_lower_words
  // Approx func name present in C: approx_count_and_lower_words
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "count_and_lower_words",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 1000>,
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 0, 1, 2>,
    thresholds = array<i32: 576>,
    decisions = array<i32: 2, 1>
  }> : () -> ()
  // Required for func_substitute:
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "count_and_lower_words"}> : () -> ()
  llvm.mlir.global internal constant @str13("\0AComputation time: %.3f ms\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("An error occurred during ranking.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Rank %d: Doc %d (Score: %.4f) - \22%s\22\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("Ranking results:\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Loaded %d documents\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("Failed to read documents from file.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("Reading documents from: %s\0A\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("Query: \22%s\22\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str5("  query: BM25 query string\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("  documents_file: File containing documents (one per line)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage: %s <documents_file> <query>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2(" .,;:!?\22'\0A\09()[]{}<>\00") {addr_space = 0 : i32}
  memref.global @B : memref<1xf64> = dense<7.500000e-01>
  memref.global @K1 : memref<1xf64> = dense<1.500000e+00>
  llvm.mlir.global internal constant @str1("Error: Cannot open file %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str0("r\00") {addr_space = 0 : i32}
  func.func @compare_tokens(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = call @strcasecmp(%arg0, %arg1) : (memref<?xi8>, memref<?xi8>) -> i32
    return %0 : i32
  }
  func.func private @strcasecmp(memref<?xi8>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @count_and_lower_words(%arg0: memref<?xi8>, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.icmp "eq" %1, %0 : !llvm.ptr
    %3 = scf.if %2 -> (i1) {
      scf.yield %true : i1
    } else {
      %5 = affine.load %arg0[0] : memref<?xi8>
      %6 = arith.extsi %5 : i8 to i32
      %7 = arith.cmpi eq, %6, %c0_i32 : i32
      scf.yield %7 : i1
    }
    %4 = scf.if %3 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %5:4 = scf.while (%arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %arg1, %arg6 = %arg0) : (i32, i32, i32, memref<?xi8>) -> (i32, i32, i32, memref<?xi8>) {
        %6 = arith.cmpi ne, %arg5, %c0_i32 : i32
        scf.condition(%6) %arg4, %arg3, %arg5, %arg6 : i32, i32, i32, memref<?xi8>
      } do {
      ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: memref<?xi8>):
        %6 = affine.load %arg6[0] : memref<?xi8>
        %7 = arith.extui %6 : i8 to i32
        %8 = func.call @tolower(%7) : (i32) -> i32
        %9 = arith.trunci %8 : i32 to i8
        affine.store %9, %arg6[0] : memref<?xi8>
        %10 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
        %11 = affine.load %10[0] : memref<?xmemref<?xi16>>
        %12 = affine.load %arg6[0] : memref<?xi8>
        %13 = arith.extui %12 : i8 to i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = memref.load %11[%14] : memref<?xi16>
        %16 = arith.extui %15 : i16 to i32
        %17 = arith.andi %16, %c8_i32 : i32
        %18 = arith.cmpi ne, %17, %c0_i32 : i32
        %19 = arith.cmpi eq, %arg4, %c0_i32 : i32
        %20 = arith.select %19, %c1_i32, %arg4 : i32
        %21 = arith.andi %18, %19 : i1
        %22 = arith.select %18, %20, %c0_i32 : i32
        %23 = scf.if %21 -> (i32) {
          %26 = arith.addi %arg3, %c1_i32 : i32
          scf.yield %26 : i32
        } else {
          scf.yield %arg3 : i32
        }
        %24 = "polygeist.subindex"(%arg6, %c1) : (memref<?xi8>, index) -> memref<?xi8>
        %25 = arith.addi %arg5, %c-1_i32 : i32
        scf.yield %22, %23, %25, %24 : i32, i32, i32, memref<?xi8>
      }
      scf.yield %5#0 : i32
    }
    return %4 : i32
  }
  func.func private @tolower(i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__ctype_b_loc() -> memref<?xmemref<?xi16>> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @approx_count_and_lower_words_1(%arg0: memref<?xi8>, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.icmp "eq" %1, %0 : !llvm.ptr
    %3 = scf.if %2 -> (i1) {
      scf.yield %true : i1
    } else {
      %5 = affine.load %arg0[0] : memref<?xi8>
      %6 = arith.extsi %5 : i8 to i32
      %7 = arith.cmpi eq, %6, %c0_i32 : i32
      scf.yield %7 : i1
    }
    %4 = scf.if %3 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %5 = arith.addi %arg1, %c1_i32 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7:3 = scf.for %arg3 = %c0 to %6 step %c2 iter_args(%arg4 = %c0_i32, %arg5 = %c0_i32, %arg6 = %arg0) -> (i32, i32, memref<?xi8>) {
        %8 = affine.load %arg6[0] : memref<?xi8>
        %9 = arith.extui %8 : i8 to i32
        %10 = func.call @tolower(%9) : (i32) -> i32
        %11 = arith.trunci %10 : i32 to i8
        affine.store %11, %arg6[0] : memref<?xi8>
        %12 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
        %13 = affine.load %12[0] : memref<?xmemref<?xi16>>
        %14 = affine.load %arg6[0] : memref<?xi8>
        %15 = arith.extui %14 : i8 to i32
        %16 = arith.index_cast %15 : i32 to index
        %17 = memref.load %13[%16] : memref<?xi16>
        %18 = arith.extui %17 : i16 to i32
        %19 = arith.andi %18, %c8_i32 : i32
        %20 = arith.cmpi ne, %19, %c0_i32 : i32
        %21 = arith.cmpi eq, %arg4, %c0_i32 : i32
        %22 = arith.select %21, %c1_i32, %arg4 : i32
        %23 = arith.andi %20, %21 : i1
        %24 = arith.select %20, %22, %c0_i32 : i32
        %25 = scf.if %23 -> (i32) {
          %27 = arith.addi %arg5, %c2_i32 : i32
          scf.yield %27 : i32
        } else {
          scf.yield %arg5 : i32
        }
        %26 = "polygeist.subindex"(%arg6, %c2) : (memref<?xi8>, index) -> memref<?xi8>
        scf.yield %24, %25, %26 : i32, i32, memref<?xi8>
      }
      scf.yield %7#1 : i32
    }
    return %4 : i32
  }
  func.func @approx_count_and_lower_words_2(%arg0: memref<?xi8>, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.icmp "eq" %1, %0 : !llvm.ptr
    %3 = scf.if %2 -> (i1) {
      scf.yield %true : i1
    } else {
      %5 = affine.load %arg0[0] : memref<?xi8>
      %6 = arith.extsi %5 : i8 to i32
      %7 = arith.cmpi eq, %6, %c0_i32 : i32
      scf.yield %7 : i1
    }
    %4 = scf.if %3 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %5 = arith.addi %arg1, %c1_i32 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7:3 = scf.for %arg3 = %c0 to %6 step %c4 iter_args(%arg4 = %c0_i32, %arg5 = %c0_i32, %arg6 = %arg0) -> (i32, i32, memref<?xi8>) {
        %8 = affine.load %arg6[0] : memref<?xi8>
        %9 = arith.extui %8 : i8 to i32
        %10 = func.call @tolower(%9) : (i32) -> i32
        %11 = arith.trunci %10 : i32 to i8
        affine.store %11, %arg6[0] : memref<?xi8>
        %12 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
        %13 = affine.load %12[0] : memref<?xmemref<?xi16>>
        %14 = affine.load %arg6[0] : memref<?xi8>
        %15 = arith.extui %14 : i8 to i32
        %16 = arith.index_cast %15 : i32 to index
        %17 = memref.load %13[%16] : memref<?xi16>
        %18 = arith.extui %17 : i16 to i32
        %19 = arith.andi %18, %c8_i32 : i32
        %20 = arith.cmpi ne, %19, %c0_i32 : i32
        %21 = arith.cmpi eq, %arg4, %c0_i32 : i32
        %22 = arith.select %21, %c1_i32, %arg4 : i32
        %23 = arith.andi %20, %21 : i1
        %24 = arith.select %20, %22, %c0_i32 : i32
        %25 = scf.if %23 -> (i32) {
          %27 = arith.addi %arg5, %c4_i32 : i32
          scf.yield %27 : i32
        } else {
          scf.yield %arg5 : i32
        }
        %26 = "polygeist.subindex"(%arg6, %c4) : (memref<?xi8>, index) -> memref<?xi8>
        scf.yield %24, %25, %26 : i32, i32, memref<?xi8>
      }
      scf.yield %7#1 : i32
    }
    return %4 : i32
  }
  func.func @calculate_idf(%arg0: i32, %arg1: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 5.000000e-01 : f64
    %0 = arith.sitofp %arg1 : i32 to f64
    %1 = arith.sitofp %arg0 : i32 to f64
    %2 = arith.subf %0, %1 : f64
    %3 = arith.addf %2, %cst_1 : f64
    %4 = arith.addf %1, %cst_1 : f64
    %5 = arith.divf %3, %4 : f64
    %6 = arith.addf %5, %cst_0 : f64
    %7 = math.log %6 : f64
    %8 = arith.cmpf ogt, %7, %cst : f64
    %9 = arith.select %8, %7, %cst : f64
    return %9 : f64
  }
  func.func @tf_count_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %7 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
      %8 = llvm.icmp "eq" %7, %2 : !llvm.ptr
      scf.yield %8 : i1
    }
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6 = scf.if %4 -> (i32) {
      scf.yield %5 : i32
    } else {
      %7 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %8 = func.call @strlen(%7) : (!llvm.ptr) -> i64
      %9 = arith.cmpi ne, %8, %c0_i64 : i64
      %10 = scf.if %9 -> (i32) {
        %11 = llvm.mlir.zero : !llvm.ptr
        %12 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
        %13 = arith.index_cast %8 : i64 to index
        %14:2 = scf.while (%arg2 = %c0_i32, %arg3 = %arg1) : (i32, memref<?xi8>) -> (i32, memref<?xi8>) {
          %15 = func.call @strstr(%arg3, %arg0) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
          %16 = "polygeist.memref2pointer"(%15) : (memref<?xi8>) -> !llvm.ptr
          %17 = llvm.icmp "ne" %16, %11 : !llvm.ptr
          scf.condition(%17) %arg2, %15 : i32, memref<?xi8>
        } do {
        ^bb0(%arg2: i32, %arg3: memref<?xi8>):
          %15 = "polygeist.memref2pointer"(%arg3) : (memref<?xi8>) -> !llvm.ptr
          %16 = llvm.icmp "eq" %15, %12 : !llvm.ptr
          %17 = scf.if %16 -> (i1) {
            scf.yield %true : i1
          } else {
            %25 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %26 = affine.load %25[0] : memref<?xmemref<?xi16>>
            %27 = llvm.getelementptr %15[-1] : (!llvm.ptr) -> !llvm.ptr, i8
            %28 = llvm.load %27 : !llvm.ptr -> i8
            %29 = arith.extui %28 : i8 to i32
            %30 = arith.index_cast %29 : i32 to index
            %31 = memref.load %26[%30] : memref<?xi16>
            %32 = arith.extui %31 : i16 to i32
            %33 = arith.andi %32, %c8_i32 : i32
            %34 = arith.cmpi eq, %33, %c0_i32 : i32
            scf.yield %34 : i1
          }
          %18 = memref.load %arg3[%13] : memref<?xi8>
          %19 = arith.extsi %18 : i8 to i32
          %20 = arith.cmpi eq, %19, %c0_i32 : i32
          %21 = scf.if %20 -> (i1) {
            scf.yield %true : i1
          } else {
            %25 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %26 = affine.load %25[0] : memref<?xmemref<?xi16>>
            %27 = memref.load %arg3[%13] : memref<?xi8>
            %28 = arith.extui %27 : i8 to i32
            %29 = arith.index_cast %28 : i32 to index
            %30 = memref.load %26[%29] : memref<?xi16>
            %31 = arith.extui %30 : i16 to i32
            %32 = arith.andi %31, %c8_i32 : i32
            %33 = arith.cmpi eq, %32, %c0_i32 : i32
            scf.yield %33 : i1
          }
          %22 = arith.andi %17, %21 : i1
          %23 = scf.if %22 -> (i32) {
            %25 = arith.addi %arg2, %c1_i32 : i32
            scf.yield %25 : i32
          } else {
            scf.yield %arg2 : i32
          }
          %24 = "polygeist.subindex"(%arg3, %13) : (memref<?xi8>, index) -> memref<?xi8>
          scf.yield %23, %24 : i32, memref<?xi8>
        }
        scf.yield %14#0 : i32
      } else {
        scf.yield %5 : i32
      }
      scf.yield %10 : i32
    }
    return %6 : i32
  }
  func.func private @strlen(!llvm.ptr) -> i64
  func.func private @strstr(memref<?xi8>, memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @read_documents_from_file(%arg0: memref<?xi8>, %arg1: memref<?xi32>) -> memref<?xmemref<?xi8>> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c4096_i32 = arith.constant 4096 : i32
    %c0_i8 = arith.constant 0 : i8
    %c10_i32 = arith.constant 10 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<4096xi8>
    %alloca_0 = memref.alloca() : memref<memref<?xmemref<?xi8>>>
    %1 = llvm.mlir.addressof @str0 : !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi8>
    %3 = call @fopen(%arg0, %2) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %4 = "polygeist.memref2pointer"(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.icmp "eq" %4, %5 : !llvm.ptr
    scf.if %6 {
      %8 = llvm.mlir.addressof @stderr : !llvm.ptr
      %9 = llvm.load %8 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %10 = "polygeist.memref2pointer"(%9) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %11 = llvm.mlir.addressof @str1 : !llvm.ptr
      %12 = llvm.getelementptr %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
      %13 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %14 = llvm.call @fprintf(%10, %12, %13) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %15 = "polygeist.pointer2memref"(%5) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
      affine.store %15, %alloca_0[] : memref<memref<?xmemref<?xi8>>>
    } else {
      %cast = memref.cast %alloca : memref<4096xi8> to memref<?xi8>
      %8 = llvm.mlir.zero : !llvm.ptr
      %9 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %19 = func.call @fgets(%cast, %c4096_i32, %3) : (memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8>
        %20 = "polygeist.memref2pointer"(%19) : (memref<?xi8>) -> !llvm.ptr
        %21 = llvm.icmp "ne" %20, %8 : !llvm.ptr
        scf.condition(%21) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        %19 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %19 : i32
      }
      func.call @rewind(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> ()
      %10 = "polygeist.typeSize"() <{source = memref<?xi8>}> : () -> index
      %11 = arith.extsi %9 : i32 to i64
      %12 = arith.index_cast %10 : index to i64
      %13 = arith.muli %11, %12 : i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = arith.divui %14, %10 : index
      %alloc = memref.alloc(%15) : memref<?xmemref<?xi8>>
      %16 = "polygeist.memref2pointer"(%alloc) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %17 = llvm.mlir.zero : !llvm.ptr
      %18 = llvm.icmp "eq" %16, %17 : !llvm.ptr
      scf.if %18 {
        %19 = func.call @fclose(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
        %20 = "polygeist.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
        affine.store %20, %alloca_0[] : memref<memref<?xmemref<?xi8>>>
      } else {
        %cast_1 = memref.cast %alloca : memref<4096xi8> to memref<?xi8>
        %19 = llvm.mlir.zero : !llvm.ptr
        %20:3 = scf.while (%arg2 = %0, %arg3 = %c0_i32, %arg4 = %true, %arg5 = %true) : (i32, i32, i1, i1) -> (i32, i1, i32) {
          %21 = func.call @fgets(%cast_1, %c4096_i32, %3) : (memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8>
          %22 = "polygeist.memref2pointer"(%21) : (memref<?xi8>) -> !llvm.ptr
          %23 = llvm.icmp "ne" %22, %19 : !llvm.ptr
          %24 = scf.if %23 -> (i1) {
            %26 = arith.cmpi slt, %arg3, %9 : i32
            scf.yield %26 : i1
          } else {
            scf.yield %false : i1
          }
          %25 = arith.andi %24, %arg5 : i1
          scf.condition(%25) %arg3, %arg4, %arg2 : i32, i1, i32
        } do {
        ^bb0(%arg2: i32, %arg3: i1, %arg4: i32):
          %21 = "polygeist.memref2pointer"(%alloca) : (memref<4096xi8>) -> !llvm.ptr
          %22 = func.call @strlen(%21) : (!llvm.ptr) -> i64
          %23 = arith.cmpi sgt, %22, %c0_i64 : i64
          scf.if %23 {
            %35 = arith.addi %22, %c-1_i64 : i64
            %36 = arith.index_cast %35 : i64 to index
            %37 = memref.load %alloca[%36] : memref<4096xi8>
            %38 = arith.extsi %37 : i8 to i32
            %39 = arith.cmpi eq, %38, %c10_i32 : i32
            scf.if %39 {
              memref.store %c0_i8, %alloca[%36] : memref<4096xi8>
            }
          }
          %24 = arith.index_cast %arg2 : i32 to index
          %25 = func.call @strdup(%cast_1) : (memref<?xi8>) -> memref<?xi8>
          memref.store %25, %alloc[%24] : memref<?xmemref<?xi8>>
          %26 = arith.index_cast %arg2 : i32 to index
          %27 = memref.load %alloc[%26] : memref<?xmemref<?xi8>>
          %28 = "polygeist.memref2pointer"(%27) : (memref<?xi8>) -> !llvm.ptr
          %29 = llvm.icmp "eq" %28, %19 : !llvm.ptr
          %30 = arith.xori %29, %true : i1
          %31 = arith.andi %30, %arg3 : i1
          %32 = arith.xori %29, %true : i1
          %33 = arith.select %29, %arg2, %arg4 : i32
          %34 = scf.if %29 -> (i32) {
            %35 = arith.index_cast %arg2 : i32 to index
            scf.for %arg5 = %c0 to %35 step %c1 {
              %38 = memref.load %alloc[%arg5] : memref<?xmemref<?xi8>>
              memref.dealloc %38 : memref<?xi8>
            }
            memref.dealloc %alloc : memref<?xmemref<?xi8>>
            %36 = func.call @fclose(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
            %37 = "polygeist.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
            affine.store %37, %alloca_0[] : memref<memref<?xmemref<?xi8>>>
            scf.yield %arg2 : i32
          } else {
            %35 = arith.addi %arg2, %c1_i32 : i32
            scf.yield %35 : i32
          }
          scf.yield %33, %34, %31, %32 : i32, i32, i1, i1
        }
        scf.if %20#1 {
          %21 = func.call @fclose(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
          affine.store %20#0, %arg1[0] : memref<?xi32>
          affine.store %alloc, %alloca_0[] : memref<memref<?xmemref<?xi8>>>
        }
      }
    }
    %7 = affine.load %alloca_0[] : memref<memref<?xmemref<?xi8>>>
    return %7 : memref<?xmemref<?xi8>>
  }
  func.func private @fopen(memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fgets(memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rewind(memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fclose(memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strdup(memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @df_contains_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
      %9 = llvm.icmp "eq" %8, %2 : !llvm.ptr
      scf.yield %9 : i1
    }
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6:2 = scf.if %4 -> (i1, i32) {
      scf.yield %false, %5 : i1, i32
    } else {
      %8 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %9 = func.call @strlen(%8) : (!llvm.ptr) -> i64
      %10 = arith.cmpi ne, %9, %c0_i64 : i64
      %11:2 = scf.if %10 -> (i1, i32) {
        %12 = arith.cmpi eq, %9, %c0_i64 : i64
        %13 = arith.select %12, %c0_i32, %5 : i32
        %14 = llvm.mlir.zero : !llvm.ptr
        %15:3 = scf.while (%arg2 = %true, %arg3 = %13, %arg4 = %true, %arg5 = %arg1) : (i1, i32, i1, memref<?xi8>) -> (i1, i32, memref<?xi8>) {
          %16 = func.call @strstr(%arg5, %arg0) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
          %17 = "polygeist.memref2pointer"(%16) : (memref<?xi8>) -> !llvm.ptr
          %18 = llvm.icmp "ne" %17, %14 : !llvm.ptr
          %19 = arith.andi %18, %arg4 : i1
          scf.condition(%19) %arg2, %arg3, %16 : i1, i32, memref<?xi8>
        } do {
        ^bb0(%arg2: i1, %arg3: i32, %arg4: memref<?xi8>):
          %16 = "polygeist.memref2pointer"(%arg4) : (memref<?xi8>) -> !llvm.ptr
          %17 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
          %18 = llvm.icmp "eq" %16, %17 : !llvm.ptr
          %19 = scf.if %18 -> (i1) {
            scf.yield %true : i1
          } else {
            %33 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %34 = affine.load %33[0] : memref<?xmemref<?xi16>>
            %35 = llvm.getelementptr %16[-1] : (!llvm.ptr) -> !llvm.ptr, i8
            %36 = llvm.load %35 : !llvm.ptr -> i8
            %37 = arith.extui %36 : i8 to i32
            %38 = arith.index_cast %37 : i32 to index
            %39 = memref.load %34[%38] : memref<?xi16>
            %40 = arith.extui %39 : i16 to i32
            %41 = arith.andi %40, %c8_i32 : i32
            %42 = arith.cmpi eq, %41, %c0_i32 : i32
            scf.yield %42 : i1
          }
          %20 = arith.extsi %19 : i1 to i32
          %21 = arith.cmpi eq, %20, %c0_i32 : i32
          %22 = arith.index_cast %9 : i64 to index
          %23 = memref.load %arg4[%22] : memref<?xi8>
          %24 = arith.extsi %23 : i8 to i32
          %25 = arith.cmpi eq, %24, %c0_i32 : i32
          %26 = scf.if %25 -> (i1) {
            scf.yield %true : i1
          } else {
            %33 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %34 = affine.load %33[0] : memref<?xmemref<?xi16>>
            %35 = memref.load %arg4[%22] : memref<?xi8>
            %36 = arith.extui %35 : i8 to i32
            %37 = arith.index_cast %36 : i32 to index
            %38 = memref.load %34[%37] : memref<?xi16>
            %39 = arith.extui %38 : i16 to i32
            %40 = arith.andi %39, %c8_i32 : i32
            %41 = arith.cmpi eq, %40, %c0_i32 : i32
            scf.yield %41 : i1
          }
          %27 = arith.extsi %26 : i1 to i32
          %28 = arith.cmpi eq, %27, %c0_i32 : i32
          %29 = arith.andi %19, %28 : i1
          %30 = arith.ori %29, %21 : i1
          %31:2 = scf.if %19 -> (i1, i32) {
            %33 = arith.cmpi eq, %27, %c0_i32 : i32
            %34 = arith.andi %33, %arg2 : i1
            %35 = arith.select %26, %c1_i32, %arg3 : i32
            scf.yield %34, %35 : i1, i32
          } else {
            scf.yield %arg2, %arg3 : i1, i32
          }
          %32 = scf.if %30 -> (memref<?xi8>) {
            %33 = arith.index_cast %9 : i64 to index
            %34 = "polygeist.subindex"(%arg4, %33) : (memref<?xi8>, index) -> memref<?xi8>
            scf.yield %34 : memref<?xi8>
          } else {
            scf.yield %arg4 : memref<?xi8>
          }
          scf.yield %31#0, %31#1, %30, %32 : i1, i32, i1, memref<?xi8>
        }
        scf.yield %15#0, %15#1 : i1, i32
      } else {
        scf.yield %false, %5 : i1, i32
      }
      scf.yield %11#0, %11#1 : i1, i32
    }
    %7 = arith.select %6#0, %c0_i32, %6#1 : i32
    return %7 : i32
  }
  func.func @calculate_df(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.icmp "eq" %2, %1 : !llvm.ptr
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %7 = "polygeist.memref2pointer"(%arg1) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %8 = llvm.icmp "eq" %7, %1 : !llvm.ptr
      scf.yield %8 : i1
    }
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6 = scf.if %4 -> (i32) {
      scf.yield %5 : i32
    } else {
      %7 = func.call @lower_dup(%arg0) : (memref<?xi8>) -> memref<?xi8>
      %8 = "polygeist.memref2pointer"(%7) : (memref<?xi8>) -> !llvm.ptr
      %9 = llvm.mlir.zero : !llvm.ptr
      %10 = llvm.icmp "eq" %8, %9 : !llvm.ptr
      %11 = scf.if %10 -> (i32) {
        scf.yield %5 : i32
      } else {
        %12 = arith.index_cast %arg2 : i32 to index
        %13 = scf.for %arg3 = %c0 to %12 step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
          %14 = memref.load %arg1[%arg3] : memref<?xmemref<?xi8>>
          %15 = func.call @df_contains_whole_word(%7, %14) : (memref<?xi8>, memref<?xi8>) -> i32
          %16 = arith.cmpi ne, %15, %c0_i32 : i32
          %17 = scf.if %16 -> (i32) {
            %18 = arith.addi %arg4, %c1_i32 : i32
            scf.yield %18 : i32
          } else {
            scf.yield %arg4 : i32
          }
          scf.yield %17 : i32
        }
        memref.dealloc %7 : memref<?xi8>
        scf.yield %13 : i32
      }
      scf.yield %11 : i32
    }
    return %6 : i32
  }
  func.func private @lower_dup(%arg0: memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.icmp "eq" %0, %1 : !llvm.ptr
    %3 = scf.if %2 -> (memref<?xi8>) {
      %4 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi8>
      scf.yield %4 : memref<?xi8>
    } else {
      %4 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %5 = func.call @strlen(%4) : (!llvm.ptr) -> i64
      %6 = arith.addi %5, %c1_i64 : i64
      %7 = arith.index_cast %6 : i64 to index
      %alloc = memref.alloc(%7) : memref<?xi8>
      %8 = "polygeist.memref2pointer"(%alloc) : (memref<?xi8>) -> !llvm.ptr
      %9 = llvm.mlir.zero : !llvm.ptr
      %10 = llvm.icmp "eq" %8, %9 : !llvm.ptr
      %11 = scf.if %10 -> (memref<?xi8>) {
        %12 = "polygeist.pointer2memref"(%9) : (!llvm.ptr) -> memref<?xi8>
        scf.yield %12 : memref<?xi8>
      } else {
        %12 = arith.index_cast %5 : i64 to index
        scf.for %arg1 = %c0 to %12 step %c1 {
          %14 = memref.load %arg0[%arg1] : memref<?xi8>
          %15 = arith.extui %14 : i8 to i32
          %16 = func.call @tolower(%15) : (i32) -> i32
          %17 = arith.trunci %16 : i32 to i8
          memref.store %17, %alloc[%arg1] : memref<?xi8>
        }
        %13 = arith.index_cast %5 : i64 to index
        memref.store %c0_i8, %alloc[%13] : memref<?xi8>
        scf.yield %alloc : memref<?xi8>
      }
      scf.yield %11 : memref<?xi8>
    }
    return %3 : memref<?xi8>
  }
  func.func @score_term_over_docs(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xf64>, %arg3: f64, %arg4: f64, %arg5: memref<?x!llvm.struct<(i32, f64)>>, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %0 = arith.index_cast %arg6 : i32 to index
    scf.for %arg8 = %c0 to %0 step %c1 {
      %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %2 = func.call @strlen(%1) : (!llvm.ptr) -> i64
      %3 = memref.load %arg1[%arg8] : memref<?xmemref<?xi8>>
      %4 = func.call @tf_count_whole_word(%arg0, %3) : (memref<?xi8>, memref<?xi8>) -> i32
      %5 = arith.sitofp %4 : i32 to f64
      %6 = memref.get_global @K1 : memref<1xf64>
      %7 = affine.load %6[0] : memref<1xf64>
      %8 = arith.addf %7, %cst_0 : f64
      %9 = arith.mulf %5, %8 : f64
      %10 = memref.get_global @B : memref<1xf64>
      %11 = affine.load %10[0] : memref<1xf64>
      %12 = arith.subf %cst_0, %11 : f64
      %13 = memref.load %arg2[%arg8] : memref<?xf64>
      %14 = arith.divf %13, %arg3 : f64
      %15 = arith.mulf %11, %14 : f64
      %16 = arith.addf %12, %15 : f64
      %17 = arith.mulf %7, %16 : f64
      %18 = arith.addf %5, %17 : f64
      %19 = arith.cmpf ogt, %18, %cst : f64
      %20 = scf.if %19 -> (f64) {
        %29 = arith.divf %9, %18 : f64
        scf.yield %29 : f64
      } else {
        scf.yield %cst : f64
      }
      %21 = arith.mulf %arg4, %20 : f64
      %22 = arith.muli %arg8, %c16 : index
      %23 = arith.index_cast %22 : index to i64
      %24 = "polygeist.memref2pointer"(%arg5) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
      %25 = llvm.getelementptr %24[%23] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %26 = llvm.getelementptr %25[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
      %27 = llvm.load %26 : !llvm.ptr -> f64
      %28 = arith.addf %27, %21 : f64
      llvm.store %28, %26 : f64, !llvm.ptr
    }
    return
  }
  func.func @approx_dup(%arg0: memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i8 = arith.constant 0 : i8
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %1 = call @strlen(%0) : (!llvm.ptr) -> i64
    %2 = arith.divsi %1, %c2_i64 : i64
    %3 = arith.addi %2, %c1_i64 : i64
    %4 = arith.index_cast %3 : i64 to index
    %alloc = memref.alloc(%4) : memref<?xi8>
    %5 = "polygeist.memref2pointer"(%alloc) : (memref<?xi8>) -> !llvm.ptr
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.icmp "eq" %5, %6 : !llvm.ptr
    %8 = scf.if %7 -> (memref<?xi8>) {
      %9 = "polygeist.pointer2memref"(%6) : (!llvm.ptr) -> memref<?xi8>
      scf.yield %9 : memref<?xi8>
    } else {
      %9:3 = scf.while (%arg1 = %c0_i32, %arg2 = %c0_i32, %arg3 = %c1_i32) : (i32, i32, i32) -> (i32, i32, i32) {
        %11 = arith.extsi %arg2 : i32 to i64
        %12 = arith.cmpi slt, %11, %2 : i64
        %13 = arith.extsi %arg1 : i32 to i64
        %14 = arith.cmpi slt, %13, %1 : i64
        %15 = arith.andi %12, %14 : i1
        scf.condition(%15) %arg2, %arg1, %arg3 : i32, i32, i32
      } do {
      ^bb0(%arg1: i32, %arg2: i32, %arg3: i32):
        %11 = arith.cmpi ne, %arg3, %c0_i32 : i32
        %12 = scf.if %11 -> (i32) {
          %16 = arith.addi %arg1, %c1_i32 : i32
          %17 = arith.index_cast %arg1 : i32 to index
          %18 = arith.index_cast %arg2 : i32 to index
          %19 = memref.load %arg0[%18] : memref<?xi8>
          memref.store %19, %alloc[%17] : memref<?xi8>
          scf.yield %16 : i32
        } else {
          scf.yield %arg1 : i32
        }
        %13 = arith.extsi %arg2 : i32 to i64
        %14 = arith.cmpi slt, %13, %1 : i64
        %15:2 = scf.if %14 -> (i32, i32) {
          %16 = arith.addi %arg2, %c1_i32 : i32
          %17 = arith.index_cast %arg2 : i32 to index
          %18 = memref.load %arg0[%17] : memref<?xi8>
          %19 = arith.extsi %18 : i8 to i32
          %20 = arith.cmpi eq, %19, %c32_i32 : i32
          %21 = scf.if %20 -> (i32) {
            %22 = arith.xori %arg3, %c1_i32 : i32
            scf.yield %22 : i32
          } else {
            scf.yield %arg3 : i32
          }
          scf.yield %16, %21 : i32, i32
        } else {
          scf.yield %arg2, %arg3 : i32, i32
        }
        scf.yield %15#0, %12, %15#1 : i32, i32, i32
      }
      %10 = arith.index_cast %9#0 : i32 to index
      memref.store %c0_i8, %alloc[%10] : memref<?xi8>
      scf.yield %alloc : memref<?xi8>
    }
    return %8 : memref<?xi8>
  }
  func.func @rank_documents_bm25(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: i32, %arg3: i32) -> memref<?x!llvm.struct<(i32, f64)>> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c16 = arith.constant 16 : index
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %alloca_0 = memref.alloca() : memref<256xmemref<?xi8>>
    %alloca_1 = memref.alloca() : memref<memref<?x!llvm.struct<(i32, f64)>>>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %4 = llvm.icmp "eq" %3, %2 : !llvm.ptr
    %5 = scf.if %4 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = "polygeist.memref2pointer"(%arg1) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %9 = llvm.icmp "eq" %8, %2 : !llvm.ptr
      scf.yield %9 : i1
    }
    %6 = scf.if %5 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = arith.cmpi sle, %arg2, %c0_i32 : i32
      scf.yield %8 : i1
    }
    scf.if %6 {
      %8 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
      affine.store %8, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
    } else {
      %8 = arith.extsi %arg2 : i32 to i64
      %9 = arith.muli %8, %c8_i64 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = arith.divui %10, %c8 : index
      %alloc = memref.alloc(%11) : memref<?xf64>
      %12 = "polygeist.typeSize"() <{source = memref<?xi8>}> : () -> index
      %13 = arith.extsi %arg2 : i32 to i64
      %14 = arith.index_cast %12 : index to i64
      %15 = arith.muli %13, %14 : i64
      %16 = arith.index_cast %15 : i64 to index
      %17 = arith.divui %16, %12 : index
      %alloc_2 = memref.alloc(%17) : memref<?xmemref<?xi8>>
      %18 = "polygeist.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
      %19 = llvm.mlir.zero : !llvm.ptr
      %20 = llvm.icmp "eq" %18, %19 : !llvm.ptr
      %21 = scf.if %20 -> (i1) {
        scf.yield %true : i1
      } else {
        %22 = "polygeist.memref2pointer"(%alloc_2) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
        %23 = llvm.icmp "eq" %22, %19 : !llvm.ptr
        scf.yield %23 : i1
      }
      scf.if %21 {
        memref.dealloc %alloc : memref<?xf64>
        memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
        %22 = "polygeist.pointer2memref"(%19) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
        affine.store %22, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
      } else {
        %22:4 = scf.while (%arg4 = %0, %arg5 = %c0_i32, %arg6 = %cst, %arg7 = %true, %arg8 = %true) : (i32, i32, f64, i1, i1) -> (f64, i1, i32, i32) {
          %23 = arith.cmpi slt, %arg5, %arg2 : i32
          %24 = arith.andi %23, %arg8 : i1
          scf.condition(%24) %arg6, %arg7, %arg4, %arg5 : f64, i1, i32, i32
        } do {
        ^bb0(%arg4: f64, %arg5: i1, %arg6: i32, %arg7: i32):
          %23 = arith.index_cast %arg7 : i32 to index
          %24 = memref.load %arg1[%23] : memref<?xmemref<?xi8>>
          %25 = func.call @strdup(%24) : (memref<?xi8>) -> memref<?xi8>
          %26 = "polygeist.memref2pointer"(%25) : (memref<?xi8>) -> !llvm.ptr
          %27 = func.call @strlen(%26) : (!llvm.ptr) -> i64
          %28 = arith.trunci %27 : i64 to i32
          %29 = "polygeist.memref2pointer"(%25) : (memref<?xi8>) -> !llvm.ptr
          %30 = llvm.mlir.zero : !llvm.ptr
          %31 = llvm.icmp "eq" %29, %30 : !llvm.ptr
          %32 = arith.xori %31, %true : i1
          %33 = arith.andi %32, %arg5 : i1
          %34 = arith.xori %31, %true : i1
          %35 = arith.select %31, %arg7, %arg6 : i32
          %36:2 = scf.if %31 -> (f64, i32) {
            %37 = arith.index_cast %arg7 : i32 to index
            scf.for %arg8 = %c0 to %37 step %c1 {
              %39 = memref.load %alloc_2[%arg8] : memref<?xmemref<?xi8>>
              memref.dealloc %39 : memref<?xi8>
            }
            memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
            memref.dealloc %alloc : memref<?xf64>
            %38 = "polygeist.pointer2memref"(%30) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
            affine.store %38, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
            scf.yield %arg4, %arg7 : f64, i32
          } else {
            %37 = arith.index_cast %arg7 : i32 to index
            %38 = func.call @count_and_lower_words(%25, %28, %28) : (memref<?xi8>, i32, i32) -> i32
            %39 = arith.sitofp %38 : i32 to f64
            memref.store %39, %alloc[%37] : memref<?xf64>
            %40 = arith.index_cast %arg7 : i32 to index
            %41 = memref.load %alloc[%40] : memref<?xf64>
            %42 = arith.addf %arg4, %41 : f64
            %43 = arith.index_cast %arg7 : i32 to index
            memref.store %25, %alloc_2[%43] : memref<?xmemref<?xi8>>
            %44 = arith.addi %arg7, %c1_i32 : i32
            scf.yield %42, %44 : f64, i32
          }
          scf.yield %35, %36#1, %36#0, %33, %34 : i32, i32, f64, i1, i1
        }
        scf.if %22#1 {
          %23 = arith.sitofp %arg2 : i32 to f64
          %24 = arith.divf %22#0, %23 : f64
          %25 = arith.extsi %arg2 : i32 to i64
          %26 = arith.muli %25, %c16_i64 : i64
          %27 = arith.index_cast %26 : i64 to index
          %28 = arith.divui %27, %c16 : index
          %alloc_3 = memref.alloc(%28) : memref<?x!llvm.struct<(i32, f64)>>
          %29 = "polygeist.memref2pointer"(%alloc_3) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
          %30 = llvm.mlir.zero : !llvm.ptr
          %31 = llvm.icmp "eq" %29, %30 : !llvm.ptr
          scf.if %31 {
            %32 = arith.index_cast %arg2 : i32 to index
            scf.for %arg4 = %c0 to %32 step %c1 {
              %34 = memref.load %alloc_2[%arg4] : memref<?xmemref<?xi8>>
              memref.dealloc %34 : memref<?xi8>
            }
            memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
            memref.dealloc %alloc : memref<?xf64>
            %33 = "polygeist.pointer2memref"(%30) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
            affine.store %33, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
          } else {
            %32 = arith.index_cast %arg2 : i32 to index
            scf.for %arg4 = %c0 to %32 step %c1 {
              %37 = arith.index_cast %arg4 : index to i32
              %38 = arith.muli %arg4, %c16 : index
              %39 = arith.index_cast %38 : index to i64
              %40 = "polygeist.memref2pointer"(%alloc_3) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
              %41 = llvm.getelementptr %40[%39] : (!llvm.ptr, i64) -> !llvm.ptr, i8
              llvm.store %37, %41 : i32, !llvm.ptr
              %42 = llvm.getelementptr %41[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
              llvm.store %cst, %42 : f64, !llvm.ptr
            }
            %33 = func.call @strdup(%arg0) : (memref<?xi8>) -> memref<?xi8>
            %34 = "polygeist.memref2pointer"(%33) : (memref<?xi8>) -> !llvm.ptr
            %35 = llvm.mlir.zero : !llvm.ptr
            %36 = llvm.icmp "eq" %34, %35 : !llvm.ptr
            scf.if %36 {
              %37 = arith.index_cast %arg2 : i32 to index
              scf.for %arg4 = %c0 to %37 step %c1 {
                %39 = memref.load %alloc_2[%arg4] : memref<?xmemref<?xi8>>
                memref.dealloc %39 : memref<?xi8>
              }
              memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
              memref.dealloc %alloc : memref<?xf64>
              memref.dealloc %alloc_3 : memref<?x!llvm.struct<(i32, f64)>>
              %38 = "polygeist.pointer2memref"(%35) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
              affine.store %38, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
            } else {
              affine.store %33, %alloca[0] : memref<1xmemref<?xi8>>
              %37 = llvm.mlir.addressof @str2 : !llvm.ptr
              %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
              %38 = "polygeist.pointer2memref"(%37) : (!llvm.ptr) -> memref<?xi8>
              %39 = llvm.mlir.zero : !llvm.ptr
              %40:5 = scf.while (%arg4 = %1, %arg5 = %0, %arg6 = %0, %arg7 = %0, %arg8 = %c0_i32, %arg9 = %true) : (f64, i32, i32, i32, i32, i1) -> (i32, f64, i32, i32, i32) {
                scf.condition(%arg9) %arg8, %arg4, %arg5, %arg6, %arg7 : i32, f64, i32, i32, i32
              } do {
              ^bb0(%arg4: i32, %arg5: f64, %arg6: i32, %arg7: i32, %arg8: i32):
                %43 = affine.load %alloca[0] : memref<1xmemref<?xi8>>
                %44 = func.call @strtok_r(%43, %38, %cast) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
                %45 = "polygeist.memref2pointer"(%44) : (memref<?xi8>) -> !llvm.ptr
                %46 = llvm.icmp "eq" %45, %39 : !llvm.ptr
                %47:6 = scf.if %46 -> (i32, i32, i1, i32, i32, f64) {
                  scf.yield %arg7, %arg8, %false, %arg4, %arg6, %arg5 : i32, i32, i1, i32, i32, f64
                } else {
                  %48 = affine.load %44[0] : memref<?xi8>
                  %49 = arith.extsi %48 : i8 to i32
                  %50 = arith.cmpi ne, %49, %c0_i32 : i32
                  %51:6 = scf.if %50 -> (i32, i32, i1, i32, i32, f64) {
                    %52:2 = scf.while (%arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %true) : (i32, i32, i1) -> (i32, i32) {
                      %61 = arith.cmpi slt, %arg9, %arg4 : i32
                      %62 = arith.andi %61, %arg11 : i1
                      scf.condition(%62) %arg9, %arg10 : i32, i32
                    } do {
                    ^bb0(%arg9: i32, %arg10: i32):
                      %61 = arith.index_cast %arg9 : i32 to index
                      %62 = memref.load %alloca_0[%61] : memref<256xmemref<?xi8>>
                      %63 = func.call @compare_tokens(%44, %62) : (memref<?xi8>, memref<?xi8>) -> i32
                      %64 = arith.cmpi eq, %63, %c0_i32 : i32
                      %65 = arith.select %64, %c1_i32, %arg10 : i32
                      %66 = arith.cmpi ne, %63, %c0_i32 : i32
                      %67 = scf.if %66 -> (i32) {
                        %68 = arith.addi %arg9, %c1_i32 : i32
                        scf.yield %68 : i32
                      } else {
                        scf.yield %arg9 : i32
                      }
                      scf.yield %67, %65, %66 : i32, i32, i1
                    }
                    %53 = arith.cmpi eq, %52#1, %c0_i32 : i32
                    %54 = arith.cmpi slt, %arg4, %c256_i32 : i32
                    %55 = arith.cmpi slt, %arg4, %c256_i32 : i32
                    %56 = arith.andi %53, %54 : i1
                    %57 = arith.cmpi ne, %52#1, %c0_i32 : i32
                    %58 = arith.andi %53, %55 : i1
                    %59 = arith.ori %58, %57 : i1
                    %60:3 = scf.if %56 -> (i32, i32, f64) {
                      %61 = arith.index_cast %arg4 : i32 to index
                      %62 = func.call @strdup(%44) : (memref<?xi8>) -> memref<?xi8>
                      memref.store %62, %alloca_0[%61] : memref<256xmemref<?xi8>>
                      %63 = arith.index_cast %arg4 : i32 to index
                      %64 = memref.load %alloca_0[%63] : memref<256xmemref<?xi8>>
                      %65 = "polygeist.memref2pointer"(%64) : (memref<?xi8>) -> !llvm.ptr
                      %66 = llvm.mlir.zero : !llvm.ptr
                      %67 = llvm.icmp "eq" %65, %66 : !llvm.ptr
                      %68:3 = scf.if %67 -> (i32, i32, f64) {
                        scf.yield %arg4, %arg6, %arg5 : i32, i32, f64
                      } else {
                        %69 = arith.addi %arg4, %c1_i32 : i32
                        %70 = func.call @calculate_df(%44, %alloc_2, %arg2) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i32
                        %71 = arith.cmpi ne, %70, %c0_i32 : i32
                        %72 = scf.if %71 -> (f64) {
                          %73 = func.call @calculate_idf(%70, %arg2) : (i32, i32) -> f64
                          %74 = func.call @lower_dup(%44) : (memref<?xi8>) -> memref<?xi8>
                          %75 = "polygeist.memref2pointer"(%74) : (memref<?xi8>) -> !llvm.ptr
                          %76 = llvm.mlir.zero : !llvm.ptr
                          %77 = llvm.icmp "eq" %75, %76 : !llvm.ptr
                          %78 = arith.xori %77, %true : i1
                          scf.if %78 {
                            %79 = arith.fptosi %73 : f64 to i32
                            func.call @score_term_over_docs(%74, %alloc_2, %alloc, %24, %73, %alloc_3, %arg2, %79) : (memref<?xi8>, memref<?xmemref<?xi8>>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i32, f64)>>, i32, i32) -> ()
                            memref.dealloc %74 : memref<?xi8>
                          }
                          scf.yield %73 : f64
                        } else {
                          scf.yield %arg5 : f64
                        }
                        scf.yield %69, %70, %72 : i32, i32, f64
                      }
                      scf.yield %68#0, %68#1, %68#2 : i32, i32, f64
                    } else {
                      scf.yield %arg4, %arg6, %arg5 : i32, i32, f64
                    }
                    scf.yield %52#0, %52#1, %59, %60#0, %60#1, %60#2 : i32, i32, i1, i32, i32, f64
                  } else {
                    scf.yield %arg7, %arg8, %true, %arg4, %arg6, %arg5 : i32, i32, i1, i32, i32, f64
                  }
                  scf.yield %51#0, %51#1, %51#2, %51#3, %51#4, %51#5 : i32, i32, i1, i32, i32, f64
                }
                scf.yield %47#5, %47#4, %47#0, %47#1, %47#3, %47#2 : f64, i32, i32, i32, i32, i1
              }
              memref.dealloc %33 : memref<?xi8>
              %41 = arith.index_cast %40#0 : i32 to index
              scf.for %arg4 = %c0 to %41 step %c1 {
                %43 = memref.load %alloca_0[%arg4] : memref<256xmemref<?xi8>>
                memref.dealloc %43 : memref<?xi8>
              }
              %42 = arith.index_cast %arg2 : i32 to index
              scf.for %arg4 = %c0 to %42 step %c1 {
                %43 = memref.load %alloc_2[%arg4] : memref<?xmemref<?xi8>>
                memref.dealloc %43 : memref<?xi8>
              }
              memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
              memref.dealloc %alloc : memref<?xf64>
              affine.store %alloc_3, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
            }
          }
        }
      }
    }
    %7 = affine.load %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
    return %7 : memref<?x!llvm.struct<(i32, f64)>>
  }
  func.func private @strtok_r(memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @approx_rank_documents_bm25_1(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: i32, %arg3: i32) -> memref<?x!llvm.struct<(i32, f64)>> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c16 = arith.constant 16 : index
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %alloca_0 = memref.alloca() : memref<256xmemref<?xi8>>
    %alloca_1 = memref.alloca() : memref<memref<?x!llvm.struct<(i32, f64)>>>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %4 = llvm.icmp "eq" %3, %2 : !llvm.ptr
    %5 = scf.if %4 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = "polygeist.memref2pointer"(%arg1) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %9 = llvm.icmp "eq" %8, %2 : !llvm.ptr
      scf.yield %9 : i1
    }
    %6 = scf.if %5 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = arith.cmpi sle, %arg2, %c0_i32 : i32
      scf.yield %8 : i1
    }
    scf.if %6 {
      %8 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
      affine.store %8, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
    } else {
      %8 = arith.extsi %arg2 : i32 to i64
      %9 = arith.muli %8, %c8_i64 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = arith.divui %10, %c8 : index
      %alloc = memref.alloc(%11) : memref<?xf64>
      %12 = "polygeist.typeSize"() <{source = memref<?xi8>}> : () -> index
      %13 = arith.extsi %arg2 : i32 to i64
      %14 = arith.index_cast %12 : index to i64
      %15 = arith.muli %13, %14 : i64
      %16 = arith.index_cast %15 : i64 to index
      %17 = arith.divui %16, %12 : index
      %alloc_2 = memref.alloc(%17) : memref<?xmemref<?xi8>>
      %18:4 = scf.while (%arg4 = %0, %arg5 = %c0_i32, %arg6 = %cst, %arg7 = %true, %arg8 = %true) : (i32, i32, f64, i1, i1) -> (f64, i1, i32, i32) {
        %19 = arith.cmpi slt, %arg5, %arg2 : i32
        %20 = arith.andi %19, %arg8 : i1
        scf.condition(%20) %arg6, %arg7, %arg4, %arg5 : f64, i1, i32, i32
      } do {
      ^bb0(%arg4: f64, %arg5: i1, %arg6: i32, %arg7: i32):
        %19 = arith.index_cast %arg7 : i32 to index
        %20 = memref.load %arg1[%19] : memref<?xmemref<?xi8>>
        %21 = func.call @approx_dup(%20) : (memref<?xi8>) -> memref<?xi8>
        %22 = "polygeist.memref2pointer"(%21) : (memref<?xi8>) -> !llvm.ptr
        %23 = func.call @strlen(%22) : (!llvm.ptr) -> i64
        %24 = arith.trunci %23 : i64 to i32
        %25 = "polygeist.memref2pointer"(%21) : (memref<?xi8>) -> !llvm.ptr
        %26 = llvm.mlir.zero : !llvm.ptr
        %27 = llvm.icmp "eq" %25, %26 : !llvm.ptr
        %28 = arith.xori %27, %true : i1
        %29 = arith.andi %28, %arg5 : i1
        %30 = arith.xori %27, %true : i1
        %31 = arith.select %27, %arg7, %arg6 : i32
        %32:2 = scf.if %27 -> (f64, i32) {
          %33 = arith.index_cast %arg7 : i32 to index
          scf.for %arg8 = %c0 to %33 step %c1 {
            %35 = memref.load %alloc_2[%arg8] : memref<?xmemref<?xi8>>
            memref.dealloc %35 : memref<?xi8>
          }
          memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
          memref.dealloc %alloc : memref<?xf64>
          %34 = "polygeist.pointer2memref"(%26) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
          affine.store %34, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
          scf.yield %arg4, %arg7 : f64, i32
        } else {
          %33 = arith.index_cast %arg7 : i32 to index
          %34 = func.call @count_and_lower_words(%21, %24, %24) : (memref<?xi8>, i32, i32) -> i32
          %35 = arith.sitofp %34 : i32 to f64
          memref.store %35, %alloc[%33] : memref<?xf64>
          %36 = arith.index_cast %arg7 : i32 to index
          %37 = memref.load %alloc[%36] : memref<?xf64>
          %38 = arith.addf %arg4, %37 : f64
          %39 = arith.index_cast %arg7 : i32 to index
          memref.store %21, %alloc_2[%39] : memref<?xmemref<?xi8>>
          %40 = arith.addi %arg7, %c1_i32 : i32
          scf.yield %38, %40 : f64, i32
        }
        scf.yield %31, %32#1, %32#0, %29, %30 : i32, i32, f64, i1, i1
      }
      scf.if %18#1 {
        %19 = arith.sitofp %arg2 : i32 to f64
        %20 = arith.divf %18#0, %19 : f64
        %21 = arith.extsi %arg2 : i32 to i64
        %22 = arith.muli %21, %c16_i64 : i64
        %23 = arith.index_cast %22 : i64 to index
        %24 = arith.divui %23, %c16 : index
        %alloc_3 = memref.alloc(%24) : memref<?x!llvm.struct<(i32, f64)>>
        %25 = arith.index_cast %arg2 : i32 to index
        scf.for %arg4 = %c0 to %25 step %c1 {
          %31 = arith.index_cast %arg4 : index to i32
          %32 = arith.muli %arg4, %c16 : index
          %33 = arith.index_cast %32 : index to i64
          %34 = "polygeist.memref2pointer"(%alloc_3) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
          %35 = llvm.getelementptr %34[%33] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          llvm.store %31, %35 : i32, !llvm.ptr
          %36 = llvm.getelementptr %35[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
          llvm.store %cst, %36 : f64, !llvm.ptr
        }
        %26 = func.call @strdup(%arg0) : (memref<?xi8>) -> memref<?xi8>
        affine.store %26, %alloca[0] : memref<1xmemref<?xi8>>
        %27 = llvm.mlir.addressof @str2 : !llvm.ptr
        %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %28 = "polygeist.pointer2memref"(%27) : (!llvm.ptr) -> memref<?xi8>
        %29 = llvm.mlir.zero : !llvm.ptr
        %30:5 = scf.while (%arg4 = %1, %arg5 = %0, %arg6 = %0, %arg7 = %0, %arg8 = %c0_i32, %arg9 = %true) : (f64, i32, i32, i32, i32, i1) -> (f64, i32, i32, i32, i32) {
          scf.condition(%arg9) %arg4, %arg5, %arg6, %arg7, %arg8 : f64, i32, i32, i32, i32
        } do {
        ^bb0(%arg4: f64, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32):
          %31 = affine.load %alloca[0] : memref<1xmemref<?xi8>>
          %32 = func.call @strtok_r(%31, %28, %cast) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
          %33 = "polygeist.memref2pointer"(%32) : (memref<?xi8>) -> !llvm.ptr
          %34 = llvm.icmp "eq" %33, %29 : !llvm.ptr
          %35:6 = scf.if %34 -> (i32, i32, i1, i32, i32, f64) {
            scf.yield %arg6, %arg7, %false, %arg8, %arg5, %arg4 : i32, i32, i1, i32, i32, f64
          } else {
            %36 = affine.load %32[0] : memref<?xi8>
            %37 = arith.extsi %36 : i8 to i32
            %38 = arith.cmpi ne, %37, %c0_i32 : i32
            %39:6 = scf.if %38 -> (i32, i32, i1, i32, i32, f64) {
              %40:2 = scf.while (%arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %true) : (i32, i32, i1) -> (i32, i32) {
                %49 = arith.cmpi slt, %arg9, %arg8 : i32
                %50 = arith.andi %49, %arg11 : i1
                scf.condition(%50) %arg9, %arg10 : i32, i32
              } do {
              ^bb0(%arg9: i32, %arg10: i32):
                %49 = arith.index_cast %arg9 : i32 to index
                %50 = memref.load %alloca_0[%49] : memref<256xmemref<?xi8>>
                %51 = func.call @compare_tokens(%32, %50) : (memref<?xi8>, memref<?xi8>) -> i32
                %52 = arith.cmpi eq, %51, %c0_i32 : i32
                %53 = arith.select %52, %c1_i32, %arg10 : i32
                %54 = arith.cmpi ne, %51, %c0_i32 : i32
                %55 = scf.if %54 -> (i32) {
                  %56 = arith.addi %arg9, %c1_i32 : i32
                  scf.yield %56 : i32
                } else {
                  scf.yield %arg9 : i32
                }
                scf.yield %55, %53, %54 : i32, i32, i1
              }
              %41 = arith.cmpi eq, %40#1, %c0_i32 : i32
              %42 = arith.cmpi slt, %arg8, %c256_i32 : i32
              %43 = arith.cmpi slt, %arg8, %c256_i32 : i32
              %44 = arith.andi %41, %42 : i1
              %45 = arith.cmpi ne, %40#1, %c0_i32 : i32
              %46 = arith.andi %41, %43 : i1
              %47 = arith.ori %46, %45 : i1
              %48:3 = scf.if %44 -> (i32, i32, f64) {
                %49 = arith.index_cast %arg8 : i32 to index
                %50 = func.call @strdup(%32) : (memref<?xi8>) -> memref<?xi8>
                memref.store %50, %alloca_0[%49] : memref<256xmemref<?xi8>>
                %51 = arith.index_cast %arg8 : i32 to index
                %52 = memref.load %alloca_0[%51] : memref<256xmemref<?xi8>>
                %53 = "polygeist.memref2pointer"(%52) : (memref<?xi8>) -> !llvm.ptr
                %54 = llvm.mlir.zero : !llvm.ptr
                %55 = llvm.icmp "eq" %53, %54 : !llvm.ptr
                %56:3 = scf.if %55 -> (i32, i32, f64) {
                  scf.yield %arg8, %arg5, %arg4 : i32, i32, f64
                } else {
                  %57 = arith.addi %arg8, %c1_i32 : i32
                  %58 = func.call @calculate_df(%32, %alloc_2, %arg2) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i32
                  %59 = arith.cmpi ne, %58, %c0_i32 : i32
                  %60 = scf.if %59 -> (f64) {
                    %61 = func.call @calculate_idf(%58, %arg2) : (i32, i32) -> f64
                    %62 = func.call @lower_dup(%32) : (memref<?xi8>) -> memref<?xi8>
                    %63 = "polygeist.memref2pointer"(%62) : (memref<?xi8>) -> !llvm.ptr
                    %64 = llvm.mlir.zero : !llvm.ptr
                    %65 = llvm.icmp "eq" %63, %64 : !llvm.ptr
                    %66 = arith.xori %65, %true : i1
                    scf.if %66 {
                      func.call @score_term_over_docs(%62, %alloc_2, %alloc, %20, %61, %alloc_3, %arg2, %arg2) : (memref<?xi8>, memref<?xmemref<?xi8>>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i32, f64)>>, i32, i32) -> ()
                      memref.dealloc %62 : memref<?xi8>
                    }
                    scf.yield %61 : f64
                  } else {
                    scf.yield %arg4 : f64
                  }
                  scf.yield %57, %58, %60 : i32, i32, f64
                }
                scf.yield %56#0, %56#1, %56#2 : i32, i32, f64
              } else {
                scf.yield %arg8, %arg5, %arg4 : i32, i32, f64
              }
              scf.yield %40#0, %40#1, %47, %48#0, %48#1, %48#2 : i32, i32, i1, i32, i32, f64
            } else {
              scf.yield %arg6, %arg7, %true, %arg8, %arg5, %arg4 : i32, i32, i1, i32, i32, f64
            }
            scf.yield %39#0, %39#1, %39#2, %39#3, %39#4, %39#5 : i32, i32, i1, i32, i32, f64
          }
          scf.yield %35#5, %35#4, %35#0, %35#1, %35#3, %35#2 : f64, i32, i32, i32, i32, i1
        }
        affine.store %alloc_3, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
      }
    }
    %7 = affine.load %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
    return %7 : memref<?x!llvm.struct<(i32, f64)>>
  }
  func.func @approx_rank_documents_bm25_2(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: i32, %arg3: i32) -> memref<?x!llvm.struct<(i32, f64)>> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c16 = arith.constant 16 : index
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : f64
    %1 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %alloca_0 = memref.alloca() : memref<256xmemref<?xi8>>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %4 = llvm.icmp "eq" %3, %2 : !llvm.ptr
    %5 = scf.if %4 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = "polygeist.memref2pointer"(%arg1) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %9 = llvm.icmp "eq" %8, %2 : !llvm.ptr
      scf.yield %9 : i1
    }
    %6 = scf.if %5 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = arith.cmpi sle, %arg2, %c0_i32 : i32
      scf.yield %8 : i1
    }
    %7 = scf.if %6 -> (memref<?x!llvm.struct<(i32, f64)>>) {
      %8 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
      scf.yield %8 : memref<?x!llvm.struct<(i32, f64)>>
    } else {
      %8 = arith.extsi %arg2 : i32 to i64
      %9 = arith.muli %8, %c8_i64 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = arith.divui %10, %c8 : index
      %alloc = memref.alloc(%11) : memref<?xf64>
      %12 = "polygeist.typeSize"() <{source = memref<?xi8>}> : () -> index
      %13 = arith.extsi %arg2 : i32 to i64
      %14 = arith.index_cast %12 : index to i64
      %15 = arith.muli %13, %14 : i64
      %16 = arith.index_cast %15 : i64 to index
      %17 = arith.divui %16, %12 : index
      %alloc_1 = memref.alloc(%17) : memref<?xmemref<?xi8>>
      %18 = arith.index_cast %arg2 : i32 to index
      %19 = scf.for %arg4 = %c0 to %18 step %c1 iter_args(%arg5 = %cst) -> (f64) {
        %32 = arith.index_cast %arg4 : index to i32
        %33 = memref.load %arg1[%arg4] : memref<?xmemref<?xi8>>
        %34 = func.call @approx_dup(%33) : (memref<?xi8>) -> memref<?xi8>
        %35 = "polygeist.memref2pointer"(%34) : (memref<?xi8>) -> !llvm.ptr
        %36 = func.call @strlen(%35) : (!llvm.ptr) -> i64
        %37 = arith.trunci %36 : i64 to i32
        %38 = arith.andi %32, %c1_i32 : i32
        %39 = arith.cmpi ne, %38, %c0_i32 : i32
        %40 = scf.if %39 -> (f64) {
          %42 = func.call @count_and_lower_words(%34, %37, %37) : (memref<?xi8>, i32, i32) -> i32
          %43 = arith.sitofp %42 : i32 to f64
          memref.store %43, %alloc[%arg4] : memref<?xf64>
          scf.yield %43 : f64
        } else {
          %42 = func.call @approx_count_and_lower_words_1(%34, %37, %37) : (memref<?xi8>, i32, i32) -> i32
          %43 = arith.sitofp %42 : i32 to f64
          memref.store %43, %alloc[%arg4] : memref<?xf64>
          scf.yield %43 : f64
        }
        %41 = arith.addf %arg5, %40 : f64
        memref.store %34, %alloc_1[%arg4] : memref<?xmemref<?xi8>>
        scf.yield %41 : f64
      }
      %20 = arith.sitofp %arg2 : i32 to f64
      %21 = arith.divf %19, %20 : f64
      %22 = arith.extsi %arg2 : i32 to i64
      %23 = arith.muli %22, %c16_i64 : i64
      %24 = arith.index_cast %23 : i64 to index
      %25 = arith.divui %24, %c16 : index
      %alloc_2 = memref.alloc(%25) : memref<?x!llvm.struct<(i32, f64)>>
      %26 = arith.index_cast %arg2 : i32 to index
      scf.for %arg4 = %c0 to %26 step %c1 {
        %32 = arith.index_cast %arg4 : index to i32
        %33 = arith.muli %arg4, %c16 : index
        %34 = arith.index_cast %33 : index to i64
        %35 = "polygeist.memref2pointer"(%alloc_2) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %36 = llvm.getelementptr %35[%34] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        llvm.store %32, %36 : i32, !llvm.ptr
        %37 = llvm.getelementptr %36[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        llvm.store %cst, %37 : f64, !llvm.ptr
      }
      %27 = func.call @strdup(%arg0) : (memref<?xi8>) -> memref<?xi8>
      affine.store %27, %alloca[0] : memref<1xmemref<?xi8>>
      %28 = llvm.mlir.addressof @str2 : !llvm.ptr
      %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
      %29 = "polygeist.pointer2memref"(%28) : (!llvm.ptr) -> memref<?xi8>
      %30 = llvm.mlir.zero : !llvm.ptr
      %31:5 = scf.while (%arg4 = %0, %arg5 = %1, %arg6 = %1, %arg7 = %1, %arg8 = %c0_i32, %arg9 = %true) : (f64, i32, i32, i32, i32, i1) -> (f64, i32, i32, i32, i32) {
        scf.condition(%arg9) %arg4, %arg5, %arg6, %arg7, %arg8 : f64, i32, i32, i32, i32
      } do {
      ^bb0(%arg4: f64, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32):
        %32 = affine.load %alloca[0] : memref<1xmemref<?xi8>>
        %33 = func.call @strtok_r(%32, %29, %cast) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
        %34 = "polygeist.memref2pointer"(%33) : (memref<?xi8>) -> !llvm.ptr
        %35 = llvm.icmp "eq" %34, %30 : !llvm.ptr
        %36:6 = scf.if %35 -> (i32, i32, i1, i32, i32, f64) {
          scf.yield %arg6, %arg7, %false, %arg8, %arg5, %arg4 : i32, i32, i1, i32, i32, f64
        } else {
          %37 = affine.load %33[0] : memref<?xi8>
          %38 = arith.extsi %37 : i8 to i32
          %39 = arith.cmpi ne, %38, %c0_i32 : i32
          %40:6 = scf.if %39 -> (i32, i32, i1, i32, i32, f64) {
            %41:2 = scf.while (%arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %true) : (i32, i32, i1) -> (i32, i32) {
              %50 = arith.cmpi slt, %arg9, %arg8 : i32
              %51 = arith.andi %50, %arg11 : i1
              scf.condition(%51) %arg9, %arg10 : i32, i32
            } do {
            ^bb0(%arg9: i32, %arg10: i32):
              %50 = arith.index_cast %arg9 : i32 to index
              %51 = memref.load %alloca_0[%50] : memref<256xmemref<?xi8>>
              %52 = func.call @compare_tokens(%33, %51) : (memref<?xi8>, memref<?xi8>) -> i32
              %53 = arith.cmpi eq, %52, %c0_i32 : i32
              %54 = arith.select %53, %c1_i32, %arg10 : i32
              %55 = arith.cmpi ne, %52, %c0_i32 : i32
              %56 = scf.if %55 -> (i32) {
                %57 = arith.addi %arg9, %c1_i32 : i32
                scf.yield %57 : i32
              } else {
                scf.yield %arg9 : i32
              }
              scf.yield %56, %54, %55 : i32, i32, i1
            }
            %42 = arith.cmpi eq, %41#1, %c0_i32 : i32
            %43 = arith.cmpi slt, %arg8, %c256_i32 : i32
            %44 = arith.cmpi slt, %arg8, %c256_i32 : i32
            %45 = arith.andi %42, %43 : i1
            %46 = arith.cmpi ne, %41#1, %c0_i32 : i32
            %47 = arith.andi %42, %44 : i1
            %48 = arith.ori %47, %46 : i1
            %49:3 = scf.if %45 -> (i32, i32, f64) {
              %50 = arith.index_cast %arg8 : i32 to index
              %51 = func.call @strdup(%33) : (memref<?xi8>) -> memref<?xi8>
              memref.store %51, %alloca_0[%50] : memref<256xmemref<?xi8>>
              %52 = arith.index_cast %arg8 : i32 to index
              %53 = memref.load %alloca_0[%52] : memref<256xmemref<?xi8>>
              %54 = "polygeist.memref2pointer"(%53) : (memref<?xi8>) -> !llvm.ptr
              %55 = llvm.mlir.zero : !llvm.ptr
              %56 = llvm.icmp "eq" %54, %55 : !llvm.ptr
              %57:3 = scf.if %56 -> (i32, i32, f64) {
                scf.yield %arg8, %arg5, %arg4 : i32, i32, f64
              } else {
                %58 = arith.addi %arg8, %c1_i32 : i32
                %59 = func.call @calculate_df(%33, %alloc_1, %arg2) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i32
                %60 = arith.cmpi ne, %59, %c0_i32 : i32
                %61 = scf.if %60 -> (f64) {
                  %62 = func.call @calculate_idf(%59, %arg2) : (i32, i32) -> f64
                  %63 = func.call @lower_dup(%33) : (memref<?xi8>) -> memref<?xi8>
                  %64 = "polygeist.memref2pointer"(%63) : (memref<?xi8>) -> !llvm.ptr
                  %65 = llvm.mlir.zero : !llvm.ptr
                  %66 = llvm.icmp "eq" %64, %65 : !llvm.ptr
                  %67 = arith.xori %66, %true : i1
                  scf.if %67 {
                    func.call @score_term_over_docs(%63, %alloc_1, %alloc, %21, %62, %alloc_2, %arg2, %59) : (memref<?xi8>, memref<?xmemref<?xi8>>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i32, f64)>>, i32, i32) -> ()
                    memref.dealloc %63 : memref<?xi8>
                  }
                  scf.yield %62 : f64
                } else {
                  scf.yield %arg4 : f64
                }
                scf.yield %58, %59, %61 : i32, i32, f64
              }
              scf.yield %57#0, %57#1, %57#2 : i32, i32, f64
            } else {
              scf.yield %arg8, %arg5, %arg4 : i32, i32, f64
            }
            scf.yield %41#0, %41#1, %48, %49#0, %49#1, %49#2 : i32, i32, i1, i32, i32, f64
          } else {
            scf.yield %arg6, %arg7, %true, %arg8, %arg5, %arg4 : i32, i32, i1, i32, i32, f64
          }
          scf.yield %40#0, %40#1, %40#2, %40#3, %40#4, %40#5 : i32, i32, i1, i32, i32, f64
        }
        scf.yield %36#5, %36#4, %36#0, %36#1, %36#3, %36#2 : f64, i32, i32, i32, i32, i1
      }
      scf.yield %alloc_2 : memref<?x!llvm.struct<(i32, f64)>>
    }
    return %7 : memref<?x!llvm.struct<(i32, f64)>>
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+06 : f64
    %cst_0 = arith.constant 1.000000e+03 : f64
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1x2xi64>
    %alloca_1 = memref.alloca() : memref<1x2xi64>
    %alloca_2 = memref.alloca() : memref<memref<?xmemref<?xi8>>>
    %alloca_3 = memref.alloca() : memref<1xi32>
    affine.store %0, %alloca_3[0] : memref<1xi32>
    %alloca_4 = memref.alloca() : memref<memref<?xi8>>
    %1 = arith.cmpi slt, %arg0, %c3_i32 : i32
    %2 = arith.cmpi sge, %arg0, %c3_i32 : i32
    %3 = arith.select %1, %c1_i32, %0 : i32
    scf.if %1 {
      %6 = llvm.mlir.addressof @stderr : !llvm.ptr
      %7 = llvm.load %6 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %8 = "polygeist.memref2pointer"(%7) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %9 = llvm.mlir.addressof @str3 : !llvm.ptr
      %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
      %11 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %12 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
      %13 = llvm.call @fprintf(%8, %10, %12) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %14 = llvm.mlir.addressof @stderr : !llvm.ptr
      %15 = llvm.load %14 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %16 = "polygeist.memref2pointer"(%15) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %17 = llvm.mlir.addressof @str4 : !llvm.ptr
      %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<60 x i8>
      %19 = llvm.call @fprintf(%16, %18) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %20 = llvm.mlir.addressof @stderr : !llvm.ptr
      %21 = llvm.load %20 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %22 = "polygeist.memref2pointer"(%21) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %23 = llvm.mlir.addressof @str5 : !llvm.ptr
      %24 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
      %25 = llvm.call @fprintf(%22, %24) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    %4:3 = scf.if %2 -> (i32, i1, i32) {
      %6 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %7 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
      affine.store %7, %alloca_4[] : memref<memref<?xi8>>
      %8 = arith.cmpi eq, %arg0, %c4_i32 : i32
      %9 = scf.if %8 -> (i32) {
        %24 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
        %25 = func.call @atoi(%24) : (memref<?xi8>) -> i32
        scf.yield %25 : i32
      } else {
        scf.yield %c3_i32 : i32
      }
      %10 = llvm.mlir.addressof @str6 : !llvm.ptr
      %11 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<13 x i8>
      %12 = "polygeist.memref2pointer"(%7) : (memref<?xi8>) -> !llvm.ptr
      %13 = llvm.call @printf(%11, %12) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %14 = llvm.mlir.addressof @str7 : !llvm.ptr
      %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<29 x i8>
      %16 = "polygeist.memref2pointer"(%6) : (memref<?xi8>) -> !llvm.ptr
      %17 = llvm.call @printf(%15, %16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %cast = memref.cast %alloca_3 : memref<1xi32> to memref<?xi32>
      %18 = func.call @read_documents_from_file(%6, %cast) : (memref<?xi8>, memref<?xi32>) -> memref<?xmemref<?xi8>>
      affine.store %18, %alloca_2[] : memref<memref<?xmemref<?xi8>>>
      %19 = "polygeist.memref2pointer"(%18) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %20 = llvm.mlir.zero : !llvm.ptr
      %21 = llvm.icmp "eq" %19, %20 : !llvm.ptr
      %22 = arith.xori %21, %true : i1
      %23 = arith.select %21, %c1_i32, %3 : i32
      scf.if %21 {
        %24 = llvm.mlir.addressof @stderr : !llvm.ptr
        %25 = llvm.load %24 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
        %26 = "polygeist.memref2pointer"(%25) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
        %27 = llvm.mlir.addressof @str8 : !llvm.ptr
        %28 = llvm.getelementptr %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<37 x i8>
        %29 = llvm.call @fprintf(%26, %28) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      }
      scf.yield %9, %22, %23 : i32, i1, i32
    } else {
      scf.yield %0, %false, %3 : i32, i1, i32
    }
    %5 = arith.select %4#1, %c0_i32, %4#2 : i32
    scf.if %4#1 {
      %6 = llvm.mlir.addressof @str9 : !llvm.ptr
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
      %8 = affine.load %alloca_3[0] : memref<1xi32>
      %9 = llvm.call @printf(%7, %8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %cast = memref.cast %alloca_1 : memref<1x2xi64> to memref<?x2xi64>
      %10 = func.call @clock_gettime(%c1_i32, %cast) : (i32, memref<?x2xi64>) -> i32
      %11 = affine.load %alloca_4[] : memref<memref<?xi8>>
      %12 = affine.load %alloca_2[] : memref<memref<?xmemref<?xi8>>>
      %13 = affine.load %alloca_3[0] : memref<1xi32>
      %14 = func.call @rank_documents_bm25(%11, %12, %13, %4#0) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32, i32) -> memref<?x!llvm.struct<(i32, f64)>>
      %cast_5 = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
      %15 = func.call @clock_gettime(%c1_i32, %cast_5) : (i32, memref<?x2xi64>) -> i32
      %16 = affine.load %alloca[0, 0] : memref<1x2xi64>
      %17 = affine.load %alloca_1[0, 0] : memref<1x2xi64>
      %18 = arith.subi %16, %17 : i64
      %19 = arith.sitofp %18 : i64 to f64
      %20 = arith.mulf %19, %cst_0 : f64
      %21 = affine.load %alloca[0, 1] : memref<1x2xi64>
      %22 = affine.load %alloca_1[0, 1] : memref<1x2xi64>
      %23 = arith.subi %21, %22 : i64
      %24 = arith.sitofp %23 : i64 to f64
      %25 = arith.divf %24, %cst : f64
      %26 = arith.addf %20, %25 : f64
      %27 = "polygeist.memref2pointer"(%14) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
      %28 = llvm.mlir.zero : !llvm.ptr
      %29 = llvm.icmp "ne" %27, %28 : !llvm.ptr
      scf.if %29 {
        %33 = llvm.mlir.addressof @str10 : !llvm.ptr
        %34 = llvm.getelementptr %33[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<18 x i8>
        %35 = llvm.call @printf(%34) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        %36 = llvm.mlir.addressof @str11 : !llvm.ptr
        %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
        %38 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
          %39 = affine.load %alloca_3[0] : memref<1xi32>
          %40 = arith.cmpi slt, %arg2, %39 : i32
          scf.condition(%40) %arg2 : i32
        } do {
        ^bb0(%arg2: i32):
          %39 = arith.index_cast %arg2 : i32 to index
          %40 = arith.muli %39, %c16 : index
          %41 = arith.index_cast %40 : index to i64
          %42 = llvm.getelementptr %27[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %43 = llvm.load %42 : !llvm.ptr -> i32
          %44 = arith.addi %arg2, %c1_i32 : i32
          %45 = llvm.getelementptr %42[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
          %46 = llvm.load %45 : !llvm.ptr -> f64
          %47 = arith.index_cast %43 : i32 to index
          %48 = memref.load %12[%47] : memref<?xmemref<?xi8>>
          %49 = "polygeist.memref2pointer"(%48) : (memref<?xi8>) -> !llvm.ptr
          %50 = llvm.call @printf(%37, %44, %43, %46, %49) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, f64, !llvm.ptr) -> i32
          %51 = arith.addi %arg2, %c1_i32 : i32
          scf.yield %51 : i32
        }
        memref.dealloc %14 : memref<?x!llvm.struct<(i32, f64)>>
      } else {
        %33 = llvm.mlir.addressof @str12 : !llvm.ptr
        %34 = llvm.getelementptr %33[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
        %35 = llvm.call @printf(%34) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      }
      %30 = llvm.mlir.addressof @str13 : !llvm.ptr
      %31 = llvm.getelementptr %30[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
      %32 = llvm.call @printf(%31, %26) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    }
    return %5 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @clock_gettime(i32, memref<?x2xi64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}
