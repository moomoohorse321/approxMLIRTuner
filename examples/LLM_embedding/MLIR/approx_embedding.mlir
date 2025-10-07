module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  // Knob — cosine similarity core (func_substitute)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "cosine_similarity_core",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 2>,
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 1>,
    decisions = array<i32: 0, 1>
  }> : () -> ()

  // Required for func_substitute
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "cosine_similarity_core"}> : () -> ()

  // Knob — ranking top-k (func_substitute)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "rank_topk",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 2>,
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 2>,
    decisions = array<i32: 0, 1>
  }> : () -> ()

  // Required for func_substitute
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "rank_topk"}> : () -> ()
  llvm.mlir.global internal constant @str10("Loaded %d documents\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Error: No documents loaded\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("Error: Failed to parse query embedding\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("[%s]\00") {addr_space = 0 : i32}
  llvm.func @sprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str6("Document embeddings should be provided via stdin\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("Query embedding should be comma-separated values\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("Usage: %s <query_embedding> [top_k]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str3("Rank %d: Doc %d (Score: %.4f) - \22%s\22\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  memref.global @query_embedding : memref<384xf32> = uninitialized
  memref.global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>> = uninitialized
  llvm.mlir.global internal constant @str2("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("|\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stdin() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  memref.global @num_documents : memref<1xi32> = dense<0>
  llvm.mlir.global internal constant @str0(",]\00") {addr_space = 0 : i32}
  func.func @parse_embedding(%arg0: memref<?xi8>, %arg1: memref<?xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c384_i32 = arith.constant 384 : i32
    %c0_i32 = arith.constant 0 : i32
    %c91_i32 = arith.constant 91 : i32
    %c1 = arith.constant 1 : index
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %0 = llvm.mlir.undef : i32
    %1 = call @strchr(%arg0, %c91_i32) : (memref<?xi8>, i32) -> memref<?xi8>
    %2 = "polygeist.memref2pointer"(%1) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.icmp "eq" %2, %3 : !llvm.ptr
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6 = scf.if %4 -> (i32) {
      scf.yield %5 : i32
    } else {
      %7 = "polygeist.subindex"(%1, %c1) : (memref<?xi8>, index) -> memref<?xi8>
      %8 = func.call @strdup(%7) : (memref<?xi8>) -> memref<?xi8>
      %9 = "polygeist.memref2pointer"(%8) : (memref<?xi8>) -> !llvm.ptr
      %10 = llvm.mlir.zero : !llvm.ptr
      %11 = llvm.icmp "eq" %9, %10 : !llvm.ptr
      %12 = scf.if %11 -> (i32) {
        scf.yield %5 : i32
      } else {
        %13 = llvm.mlir.zero : !llvm.ptr
        %14 = "polygeist.pointer2memref"(%13) : (!llvm.ptr) -> memref<?xi8>
        affine.store %14, %alloca[0] : memref<1xmemref<?xi8>>
        %15 = llvm.mlir.addressof @str0 : !llvm.ptr
        %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %16 = "polygeist.pointer2memref"(%15) : (!llvm.ptr) -> memref<?xi8>
        %17 = func.call @strtok_r(%8, %16, %cast) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
        %18 = llvm.mlir.zero : !llvm.ptr
        %19 = "polygeist.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi8>
        %20 = llvm.mlir.addressof @str0 : !llvm.ptr
        %cast_0 = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
        %22:2 = scf.while (%arg2 = %c0_i32, %arg3 = %17) : (i32, memref<?xi8>) -> (i32, memref<?xi8>) {
          %25 = "polygeist.memref2pointer"(%arg3) : (memref<?xi8>) -> !llvm.ptr
          %26 = llvm.icmp "ne" %25, %18 : !llvm.ptr
          %27 = arith.cmpi slt, %arg2, %c384_i32 : i32
          %28 = arith.andi %26, %27 : i1
          scf.condition(%28) %arg2, %arg3 : i32, memref<?xi8>
        } do {
        ^bb0(%arg2: i32, %arg3: memref<?xi8>):
          %25 = arith.index_cast %arg2 : i32 to index
          %26 = func.call @atof(%arg3) : (memref<?xi8>) -> f64
          %27 = arith.truncf %26 : f64 to f32
          memref.store %27, %arg1[%25] : memref<?xf32>
          %28 = arith.addi %arg2, %c1_i32 : i32
          %29 = func.call @strtok_r(%19, %21, %cast_0) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
          scf.yield %28, %29 : i32, memref<?xi8>
        }
        memref.dealloc %8 : memref<?xi8>
        %23 = arith.cmpi eq, %22#0, %c384_i32 : i32
        %24 = arith.extui %23 : i1 to i32
        scf.yield %24 : i32
      }
      scf.yield %12 : i32
    }
    return %6 : i32
  }
  func.func private @strchr(memref<?xi8>, i32) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strdup(memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strtok_r(memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atof(memref<?xi8>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @load_document_embeddings() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1800 = arith.constant 1800 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %c255_i64 = arith.constant 255 : i64
    %c1_i64 = arith.constant 1 : i64
    %false = arith.constant false
    %c100000_i32 = arith.constant 100000 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %alloca = memref.alloca() : memref<100000xi8>
    %0 = memref.get_global @num_documents : memref<1xi32>
    affine.store %c0_i32, %0[0] : memref<1xi32>
    %cast = memref.cast %alloca : memref<100000xi8> to memref<?xi8>
    %1 = llvm.mlir.addressof @stdin : !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    scf.while : () -> () {
      %5 = llvm.load %1 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %6 = func.call @fgets(%cast, %c100000_i32, %5) : (memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8>
      %7 = "polygeist.memref2pointer"(%6) : (memref<?xi8>) -> !llvm.ptr
      %8 = llvm.icmp "ne" %7, %2 : !llvm.ptr
      %9 = scf.if %8 -> (i1) {
        %10 = memref.get_global @num_documents : memref<1xi32>
        %11 = affine.load %10[0] : memref<1xi32>
        %12 = arith.cmpi slt, %11, %c100000_i32 : i32
        scf.if %12 {
          %13 = "polygeist.memref2pointer"(%alloca) : (memref<100000xi8>) -> !llvm.ptr
          %14 = func.call @strlen(%13) : (!llvm.ptr) -> i64
          %15 = arith.cmpi ugt, %14, %c1_i64 : i64
          scf.if %15 {
            %16 = llvm.mlir.addressof @str1 : !llvm.ptr
            %17 = "polygeist.pointer2memref"(%16) : (!llvm.ptr) -> memref<?xi8>
            %18 = func.call @strtok(%cast, %17) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
            %19 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi8>
            %20 = llvm.mlir.addressof @str1 : !llvm.ptr
            %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
            %22 = func.call @strtok(%19, %21) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
            %23 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi8>
            %24 = llvm.mlir.addressof @str2 : !llvm.ptr
            %25 = "polygeist.pointer2memref"(%24) : (!llvm.ptr) -> memref<?xi8>
            %26 = func.call @strtok(%23, %25) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
            %27 = "polygeist.memref2pointer"(%18) : (memref<?xi8>) -> !llvm.ptr
            %28 = llvm.icmp "eq" %27, %2 : !llvm.ptr
            %29 = scf.if %28 -> (i1) {
              scf.yield %true : i1
            } else {
              %32 = "polygeist.memref2pointer"(%22) : (memref<?xi8>) -> !llvm.ptr
              %33 = llvm.icmp "eq" %32, %2 : !llvm.ptr
              scf.yield %33 : i1
            }
            %30 = scf.if %29 -> (i1) {
              scf.yield %true : i1
            } else {
              %32 = "polygeist.memref2pointer"(%26) : (memref<?xi8>) -> !llvm.ptr
              %33 = llvm.icmp "eq" %32, %2 : !llvm.ptr
              scf.yield %33 : i1
            }
            %31 = arith.xori %30, %true : i1
            scf.if %31 {
              %32 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
              %33 = memref.get_global @num_documents : memref<1xi32>
              %34 = affine.load %33[0] : memref<1xi32>
              %35 = arith.index_cast %34 : i32 to index
              %36 = arith.muli %35, %c1800 : index
              %37 = arith.index_cast %36 : index to i64
              %38 = "polygeist.memref2pointer"(%32) : (memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
              %39 = llvm.getelementptr %38[%37] : (!llvm.ptr, i64) -> !llvm.ptr, i8
              %40 = func.call @atoi(%18) : (memref<?xi8>) -> i32
              llvm.store %40, %39 : i32, !llvm.ptr
              %41 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
              %42 = memref.get_global @num_documents : memref<1xi32>
              %43 = affine.load %42[0] : memref<1xi32>
              %44 = arith.index_cast %43 : i32 to index
              %45 = arith.muli %44, %c1800 : index
              %46 = arith.index_cast %45 : index to i64
              %47 = "polygeist.memref2pointer"(%41) : (memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
              %48 = llvm.getelementptr %47[%46] : (!llvm.ptr, i64) -> !llvm.ptr, i8
              %49 = llvm.getelementptr %48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
              %50 = "polygeist.pointer2memref"(%49) : (!llvm.ptr) -> memref<?xi8>
              %51 = func.call @strncpy(%50, %22, %c255_i64) : (memref<?xi8>, memref<?xi8>, i64) -> memref<?xi8>
              %52 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
              %53 = memref.get_global @num_documents : memref<1xi32>
              %54 = affine.load %53[0] : memref<1xi32>
              %55 = arith.index_cast %54 : i32 to index
              %56 = arith.muli %55, %c1800 : index
              %57 = arith.index_cast %56 : index to i64
              %58 = "polygeist.memref2pointer"(%52) : (memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
              %59 = llvm.getelementptr %58[%57] : (!llvm.ptr, i64) -> !llvm.ptr, i8
              %60 = llvm.getelementptr %59[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
              %61 = llvm.getelementptr %60[255] : (!llvm.ptr) -> !llvm.ptr, i8
              llvm.store %c0_i8, %61 : i8, !llvm.ptr
              %62 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
              %63 = memref.get_global @num_documents : memref<1xi32>
              %64 = affine.load %63[0] : memref<1xi32>
              %65 = arith.index_cast %64 : i32 to index
              %66 = arith.muli %65, %c1800 : index
              %67 = arith.index_cast %66 : index to i64
              %68 = "polygeist.memref2pointer"(%62) : (memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
              %69 = llvm.getelementptr %68[%67] : (!llvm.ptr, i64) -> !llvm.ptr, i8
              %70 = llvm.getelementptr %69[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
              %71 = "polygeist.pointer2memref"(%70) : (!llvm.ptr) -> memref<?xf32>
              %72 = func.call @parse_embedding(%26, %71) : (memref<?xi8>, memref<?xf32>) -> i32
              %73 = arith.cmpi ne, %72, %c0_i32 : i32
              scf.if %73 {
                %74 = affine.load %63[0] : memref<1xi32>
                %75 = arith.addi %74, %c1_i32 : i32
                affine.store %75, %63[0] : memref<1xi32>
              }
            }
          }
        }
        scf.yield %12 : i1
      } else {
        scf.yield %false : i1
      }
      scf.condition(%9)
    } do {
      scf.yield
    }
    %3 = memref.get_global @num_documents : memref<1xi32>
    %4 = affine.load %3[0] : memref<1xi32>
    return %4 : i32
  }
  func.func private @fgets(memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strlen(!llvm.ptr) -> i64
  func.func private @strtok(memref<?xi8>, memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strncpy(memref<?xi8>, memref<?xi8>, i64) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @cosine_similarity_core(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f32
    %0:3 = scf.for %arg3 = %c0 to %c384 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %cst) -> (f32, f32, f32) {
      %6 = memref.load %arg0[%arg3] : memref<?xf32>
      %7 = memref.load %arg1[%arg3] : memref<?xf32>
      %8 = arith.mulf %6, %7 : f32
      %9 = arith.addf %arg6, %8 : f32
      %10 = arith.mulf %6, %6 : f32
      %11 = arith.addf %arg5, %10 : f32
      %12 = arith.mulf %7, %7 : f32
      %13 = arith.addf %arg4, %12 : f32
      scf.yield %13, %11, %9 : f32, f32, f32
    }
    %1 = math.sqrt %0#1 : f32
    %2 = math.sqrt %0#0 : f32
    %3 = arith.cmpf oeq, %1, %cst : f32
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %6 = arith.cmpf oeq, %2, %cst : f32
      scf.yield %6 : i1
    }
    %5 = scf.if %4 -> (f32) {
      scf.yield %cst : f32
    } else {
      %6 = arith.mulf %1, %2 : f32
      %7 = arith.divf %0#2, %6 : f32
      scf.yield %7 : f32
    }
    return %5 : f32
  }
  func.func @approx_cosine_similarity_core(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0:3 = scf.for %arg3 = %c0 to %c384 step %c2 iter_args(%arg4 = %cst_0, %arg5 = %cst_0, %arg6 = %cst_0) -> (f32, f32, f32) {
      %9 = memref.load %arg0[%arg3] : memref<?xf32>
      %10 = memref.load %arg1[%arg3] : memref<?xf32>
      %11 = arith.mulf %9, %10 : f32
      %12 = arith.addf %arg6, %11 : f32
      %13 = arith.mulf %9, %9 : f32
      %14 = arith.addf %arg5, %13 : f32
      %15 = arith.mulf %10, %10 : f32
      %16 = arith.addf %arg4, %15 : f32
      scf.yield %16, %14, %12 : f32, f32, f32
    }
    %1 = arith.mulf %0#2, %cst : f32
    %2 = arith.mulf %0#1, %cst : f32
    %3 = math.sqrt %2 : f32
    %4 = arith.mulf %0#0, %cst : f32
    %5 = math.sqrt %4 : f32
    %6 = arith.cmpf oeq, %3, %cst_0 : f32
    %7 = scf.if %6 -> (i1) {
      scf.yield %true : i1
    } else {
      %9 = arith.cmpf oeq, %5, %cst_0 : f32
      scf.yield %9 : i1
    }
    %8 = scf.if %7 -> (f32) {
      scf.yield %cst_0 : f32
    } else {
      %9 = arith.mulf %3, %5 : f32
      %10 = arith.divf %1, %9 : f32
      scf.yield %10 : f32
    }
    return %8 : f32
  }
  func.func @compute_similarities_with_state(%arg0: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1800 = arith.constant 1800 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @num_documents : memref<1xi32>
    %1 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
    %2 = "polygeist.memref2pointer"(%1) : (memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
    %3 = memref.get_global @query_embedding : memref<384xf32>
    %cast = memref.cast %3 : memref<384xf32> to memref<?xf32>
    %4 = scf.while (%arg1 = %c0_i32) : (i32) -> i32 {
      %5 = affine.load %0[0] : memref<1xi32>
      %6 = arith.cmpi slt, %arg1, %5 : i32
      scf.condition(%6) %arg1 : i32
    } do {
    ^bb0(%arg1: i32):
      %5 = arith.index_cast %arg1 : i32 to index
      %6 = arith.muli %5, %c1800 : index
      %7 = arith.index_cast %6 : index to i64
      %8 = llvm.getelementptr %2[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %9 = llvm.getelementptr %8[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
      %10 = llvm.getelementptr %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
      %11 = "polygeist.pointer2memref"(%10) : (!llvm.ptr) -> memref<?xf32>
      %12 = func.call @cosine_similarity_core(%cast, %11, %arg0) : (memref<?xf32>, memref<?xf32>, i32) -> f32
      llvm.store %12, %9 : f32, !llvm.ptr
      %13 = arith.addi %arg1, %c1_i32 : i32
      scf.yield %13 : i32
    }
    return
  }
  func.func @compare_docs_desc(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.getelementptr %1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
    %4 = llvm.load %3 : !llvm.ptr -> f32
    %5 = llvm.getelementptr %2[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = arith.cmpf ogt, %4, %6 : f32
    %8 = arith.select %7, %c-1_i32, %0 : i32
    %9:2 = scf.if %7 -> (i1, i32) {
      scf.yield %false, %8 : i1, i32
    } else {
      %11 = llvm.getelementptr %1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
      %12 = llvm.load %11 : !llvm.ptr -> f32
      %13 = llvm.getelementptr %2[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
      %14 = llvm.load %13 : !llvm.ptr -> f32
      %15 = arith.cmpf olt, %12, %14 : f32
      %16 = arith.xori %15, %true : i1
      %17 = arith.select %15, %c1_i32, %8 : i32
      scf.yield %16, %17 : i1, i32
    }
    %10 = arith.select %9#0, %c0_i32, %9#1 : i32
    return %10 : i32
  }
  func.func @rank_topk(%arg0: i32, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1800_i64 = arith.constant 1800 : i64
    %0 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
    %1 = "polygeist.memref2pointer"(%0) : (memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi8>
    %3 = memref.get_global @num_documents : memref<1xi32>
    %4 = affine.load %3[0] : memref<1xi32>
    %5 = arith.extsi %4 : i32 to i64
    %6 = "polygeist.get_func"() <{name = @compare_docs_desc}> : () -> !llvm.ptr
    %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr) -> memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>
    call @qsort(%2, %5, %c1800_i64, %7) : (memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) -> ()
    return
  }
  func.func private @qsort(memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @approx_rank_topk(%arg0: i32, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1800_i64 = arith.constant 1800 : i64
    %c1800 = arith.constant 1800 : index
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
    %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
    scf.if %0 {
      %1 = memref.get_global @num_documents : memref<1xi32>
      %2 = affine.load %1[0] : memref<1xi32>
      %3 = arith.cmpi sgt, %arg0, %2 : i32
      %4 = arith.select %3, %2, %arg0 : i32
      %5 = arith.extsi %4 : i32 to i64
      %6 = arith.muli %5, %c1800_i64 : i64
      %7 = arith.index_cast %6 : i64 to index
      %8 = arith.divui %7, %c1800 : index
      %alloc = memref.alloc(%8) : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
      %9 = memref.get_global @num_documents : memref<1xi32>
      %10 = affine.load %9[0] : memref<1xi32>
      %11 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
      %12 = arith.index_cast %10 : i32 to index
      %13 = scf.for %arg2 = %c0 to %12 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
        %15 = memref.load %11[%arg2] : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
        affine.store %15, %alloca[0] : memref<1x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
        %16 = scf.while (%arg4 = %arg3) : (i32) -> i32 {
          %19 = arith.cmpi sgt, %arg4, %c0_i32 : i32
          %20:2 = scf.if %19 -> (i1, i32) {
            %21 = arith.addi %arg4, %c-1_i32 : i32
            %22 = arith.index_cast %21 : i32 to index
            %23 = arith.muli %22, %c1800 : index
            %24 = arith.index_cast %23 : index to i64
            %25 = "polygeist.memref2pointer"(%alloc) : (memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
            %26 = llvm.getelementptr %25[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %27 = llvm.getelementptr %26[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
            %28 = llvm.load %27 : !llvm.ptr -> f32
            %29 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
            %30 = llvm.getelementptr %29[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
            %31 = llvm.load %30 : !llvm.ptr -> f32
            %32 = arith.cmpf olt, %28, %31 : f32
            %33 = scf.if %32 -> (i32) {
              %34 = arith.cmpi slt, %arg4, %4 : i32
              scf.if %34 {
                %36 = arith.index_cast %arg4 : i32 to index
                %37 = arith.addi %arg4, %c-1_i32 : i32
                %38 = arith.index_cast %37 : i32 to index
                %39 = memref.load %alloc[%38] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
                memref.store %39, %alloc[%36] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
              }
              %35 = arith.addi %arg4, %c-1_i32 : i32
              scf.yield %35 : i32
            } else {
              scf.yield %arg4 : i32
            }
            scf.yield %32, %33 : i1, i32
          } else {
            scf.yield %false, %arg4 : i1, i32
          }
          scf.condition(%20#0) %20#1 : i32
        } do {
        ^bb0(%arg4: i32):
          scf.yield %arg4 : i32
        }
        %17 = arith.cmpi slt, %arg3, %4 : i32
        %18 = scf.if %17 -> (i32) {
          %19 = arith.cmpi slt, %16, %arg3 : i32
          scf.if %19 {
            %21 = arith.index_cast %16 : i32 to index
            memref.store %15, %alloc[%21] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
          } else {
            %21 = arith.index_cast %arg3 : i32 to index
            memref.store %15, %alloc[%21] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
          }
          %20 = arith.addi %arg3, %c1_i32 : i32
          scf.yield %20 : i32
        } else {
          %19 = arith.cmpi slt, %16, %4 : i32
          scf.if %19 {
            %20 = arith.addi %4, %c-1_i32 : i32
            %21 = arith.index_cast %20 : i32 to index
            memref.store %15, %alloc[%21] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
            %22 = arith.addi %4, %c-1_i32 : i32
            %23 = scf.while (%arg4 = %22) : (i32) -> i32 {
              %24 = arith.cmpi sgt, %arg4, %c0_i32 : i32
              %25:2 = scf.if %24 -> (i1, i32) {
                %26 = arith.addi %arg4, %c-1_i32 : i32
                %27 = arith.index_cast %26 : i32 to index
                %28 = arith.muli %27, %c1800 : index
                %29 = arith.index_cast %28 : index to i64
                %30 = "polygeist.memref2pointer"(%alloc) : (memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
                %31 = llvm.getelementptr %30[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                %32 = llvm.getelementptr %31[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
                %33 = llvm.load %32 : !llvm.ptr -> f32
                %34 = arith.index_cast %arg4 : i32 to index
                %35 = arith.muli %34, %c1800 : index
                %36 = arith.index_cast %35 : index to i64
                %37 = llvm.getelementptr %30[%36] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                %38 = llvm.getelementptr %37[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
                %39 = llvm.load %38 : !llvm.ptr -> f32
                %40 = arith.cmpf olt, %33, %39 : f32
                %41 = scf.if %40 -> (i32) {
                  %42 = arith.addi %arg4, %c-1_i32 : i32
                  %43 = arith.index_cast %42 : i32 to index
                  %44 = memref.load %alloc[%43] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
                  %45 = arith.index_cast %arg4 : i32 to index
                  %46 = memref.load %alloc[%45] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
                  memref.store %46, %alloc[%43] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
                  memref.store %44, %alloc[%45] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
                  %47 = arith.addi %arg4, %c-1_i32 : i32
                  scf.yield %47 : i32
                } else {
                  scf.yield %arg4 : i32
                }
                scf.yield %40, %41 : i1, i32
              } else {
                scf.yield %false, %arg4 : i1, i32
              }
              scf.condition(%25#0) %25#1 : i32
            } do {
            ^bb0(%arg4: i32):
              scf.yield %arg4 : i32
            }
          }
          scf.yield %arg3 : i32
        }
        scf.yield %18 : i32
      }
      %14 = arith.index_cast %4 : i32 to index
      scf.for %arg2 = %c0 to %14 step %c1 {
        %15 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
        %16 = memref.load %alloc[%arg2] : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
        memref.store %16, %15[%arg2] : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
      }
      memref.dealloc %alloc : memref<?x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
    }
    return
  }
  func.func @output_top_k(%arg0: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1800 = arith.constant 1800 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = memref.get_global @num_documents : memref<1xi32>
    %1 = affine.load %0[0] : memref<1xi32>
    %2 = arith.cmpi slt, %arg0, %1 : i32
    %3 = arith.select %2, %arg0, %1 : i32
    %4 = arith.index_cast %3 : i32 to index
    scf.for %arg1 = %c0 to %4 step %c1 {
      %5 = arith.index_cast %arg1 : index to i32
      %6 = llvm.mlir.addressof @str3 : !llvm.ptr
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %8 = arith.addi %5, %c1_i32 : i32
      %9 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>
      %10 = arith.muli %arg1, %c1800 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = "polygeist.memref2pointer"(%9) : (memref<100000x!llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>>) -> !llvm.ptr
      %13 = llvm.getelementptr %12[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %14 = llvm.load %13 : !llvm.ptr -> i32
      %15 = llvm.getelementptr %13[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
      %16 = llvm.load %15 : !llvm.ptr -> f32
      %17 = arith.extf %16 : f32 to f64
      %18 = llvm.getelementptr %13[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<384 x f32>, array<256 x i8>, f32)>
      %19 = llvm.call @printf(%7, %8, %14, %17, %18) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, f64, !llvm.ptr) -> i32
    }
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c3_i64 = arith.constant 3 : i64
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = llvm.mlir.undef : i32
    %1 = arith.cmpi slt, %arg0, %c2_i32 : i32
    %2 = arith.cmpi sge, %arg0, %c2_i32 : i32
    %3 = arith.select %1, %c1_i32, %0 : i32
    scf.if %1 {
      %6 = llvm.mlir.addressof @stderr : !llvm.ptr
      %7 = llvm.load %6 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %8 = "polygeist.memref2pointer"(%7) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %9 = llvm.mlir.addressof @str4 : !llvm.ptr
      %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<37 x i8>
      %11 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %12 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
      %13 = llvm.call @fprintf(%8, %10, %12) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %14 = llvm.mlir.addressof @stderr : !llvm.ptr
      %15 = llvm.load %14 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %16 = "polygeist.memref2pointer"(%15) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %17 = llvm.mlir.addressof @str5 : !llvm.ptr
      %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<50 x i8>
      %19 = llvm.call @fprintf(%16, %18) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %20 = llvm.mlir.addressof @stderr : !llvm.ptr
      %21 = llvm.load %20 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %22 = "polygeist.memref2pointer"(%21) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %23 = llvm.mlir.addressof @str6 : !llvm.ptr
      %24 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<50 x i8>
      %25 = llvm.call @fprintf(%22, %24) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    %4:3 = scf.if %2 -> (i32, i1, i32) {
      %6 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %7 = arith.cmpi sgt, %arg0, %c2_i32 : i32
      %8 = scf.if %7 -> (i32) {
        %24 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %25 = func.call @atoi(%24) : (memref<?xi8>) -> i32
        scf.yield %25 : i32
      } else {
        scf.yield %c10_i32 : i32
      }
      %9 = "polygeist.memref2pointer"(%6) : (memref<?xi8>) -> !llvm.ptr
      %10 = func.call @strlen(%9) : (!llvm.ptr) -> i64
      %11 = arith.addi %10, %c3_i64 : i64
      %12 = arith.index_cast %11 : i64 to index
      %alloca = memref.alloca(%12) : memref<?xi8>
      %13 = "polygeist.memref2pointer"(%alloca) : (memref<?xi8>) -> !llvm.ptr
      %14 = llvm.mlir.addressof @str7 : !llvm.ptr
      %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
      %16 = "polygeist.memref2pointer"(%6) : (memref<?xi8>) -> !llvm.ptr
      %17 = llvm.call @sprintf(%13, %15, %16) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %18 = memref.get_global @query_embedding : memref<384xf32>
      %cast = memref.cast %18 : memref<384xf32> to memref<?xf32>
      %19 = func.call @parse_embedding(%alloca, %cast) : (memref<?xi8>, memref<?xf32>) -> i32
      %20 = arith.cmpi eq, %19, %c0_i32 : i32
      %21 = arith.cmpi ne, %19, %c0_i32 : i32
      %22 = arith.select %20, %c1_i32, %3 : i32
      scf.if %20 {
        %24 = llvm.mlir.addressof @stderr : !llvm.ptr
        %25 = llvm.load %24 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
        %26 = "polygeist.memref2pointer"(%25) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
        %27 = llvm.mlir.addressof @str8 : !llvm.ptr
        %28 = llvm.getelementptr %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<40 x i8>
        %29 = llvm.call @fprintf(%26, %28) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      }
      %23:2 = scf.if %21 -> (i1, i32) {
        %24 = func.call @load_document_embeddings() : () -> i32
        %25 = arith.cmpi eq, %24, %c0_i32 : i32
        %26 = arith.cmpi ne, %24, %c0_i32 : i32
        %27 = arith.select %25, %c1_i32, %22 : i32
        scf.if %25 {
          %28 = llvm.mlir.addressof @stderr : !llvm.ptr
          %29 = llvm.load %28 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
          %30 = "polygeist.memref2pointer"(%29) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
          %31 = llvm.mlir.addressof @str9 : !llvm.ptr
          %32 = llvm.getelementptr %31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
          %33 = llvm.call @fprintf(%30, %32) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        }
        scf.yield %26, %27 : i1, i32
      } else {
        scf.yield %false, %3 : i1, i32
      }
      scf.yield %8, %23#0, %23#1 : i32, i1, i32
    } else {
      scf.yield %0, %false, %3 : i32, i1, i32
    }
    %5 = arith.select %4#1, %c0_i32, %4#2 : i32
    scf.if %4#1 {
      %6 = llvm.mlir.addressof @str10 : !llvm.ptr
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
      %8 = memref.get_global @num_documents : memref<1xi32>
      %9 = affine.load %8[0] : memref<1xi32>
      %10 = llvm.call @printf(%7, %9) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @compute_similarities_with_state(%c0_i32) : (i32) -> ()
      func.call @rank_topk(%4#0, %c0_i32) : (i32, i32) -> ()
      func.call @output_top_k(%4#0) : (i32) -> ()
    }
    return %5 : i32
  }
}
