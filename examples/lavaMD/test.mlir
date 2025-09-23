module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str13("ERROR: open result.txt\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("Results written to result.txt\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("ERROR: data write\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("ERROR: header write\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("wb\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("result.txt\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("Total execution time: %f seconds\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: OOM\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("Configuration: boxes1d = %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("Usage: %s -boxes1d <number>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("ERROR: Usage: %s -boxes1d <number>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("ERROR: -boxes1d > 0\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("-boxes1d\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("WG size of kernel = %d\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @__internal_pair_interaction(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>, %arg6: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = affine.load %arg3[symbol(%0), 0] : memref<?x4xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = affine.load %arg3[symbol(%2), 0] : memref<?x4xf64>
    %4 = arith.addf %1, %3 : f64
    %5 = affine.load %arg3[symbol(%0), 1] : memref<?x4xf64>
    %6 = affine.load %arg3[symbol(%2), 1] : memref<?x4xf64>
    %7 = arith.mulf %5, %6 : f64
    %8 = affine.load %arg3[symbol(%0), 2] : memref<?x4xf64>
    %9 = affine.load %arg3[symbol(%2), 2] : memref<?x4xf64>
    %10 = arith.mulf %8, %9 : f64
    %11 = arith.addf %7, %10 : f64
    %12 = affine.load %arg3[symbol(%0), 3] : memref<?x4xf64>
    %13 = affine.load %arg3[symbol(%2), 3] : memref<?x4xf64>
    %14 = arith.mulf %12, %13 : f64
    %15 = arith.addf %11, %14 : f64
    %16 = arith.subf %4, %15 : f64
    %17 = arith.cmpf olt, %16, %cst_0 : f64
    %18 = arith.select %17, %cst_0, %16 : f64
    %19 = arith.mulf %arg2, %18 : f64
    %20 = arith.negf %19 : f64
    %21 = math.exp %20 : f64
    %22 = arith.mulf %21, %cst : f64
    %23 = arith.index_cast %arg0 : i32 to index
    %24 = affine.load %arg3[symbol(%23), 1] : memref<?x4xf64>
    %25 = arith.index_cast %arg1 : i32 to index
    %26 = affine.load %arg3[symbol(%25), 1] : memref<?x4xf64>
    %27 = arith.subf %24, %26 : f64
    %28 = arith.index_cast %arg0 : i32 to index
    %29 = affine.load %arg3[symbol(%28), 2] : memref<?x4xf64>
    %30 = arith.index_cast %arg1 : i32 to index
    %31 = affine.load %arg3[symbol(%30), 2] : memref<?x4xf64>
    %32 = arith.subf %29, %31 : f64
    %33 = arith.index_cast %arg0 : i32 to index
    %34 = affine.load %arg3[symbol(%33), 3] : memref<?x4xf64>
    %35 = arith.index_cast %arg1 : i32 to index
    %36 = affine.load %arg3[symbol(%35), 3] : memref<?x4xf64>
    %37 = arith.subf %34, %36 : f64
    %38 = arith.index_cast %arg1 : i32 to index
    %39 = affine.load %arg4[symbol(%38)] : memref<?xf64>
    %40 = arith.mulf %39, %21 : f64
    %41 = affine.load %arg5[0, 0] : memref<?x4xf64>
    %42 = arith.addf %41, %40 : f64
    affine.store %42, %arg5[0, 0] : memref<?x4xf64>
    %43 = arith.index_cast %arg1 : i32 to index
    %44 = affine.load %arg4[symbol(%43)] : memref<?xf64>
    %45 = arith.mulf %22, %27 : f64
    %46 = arith.mulf %44, %45 : f64
    %47 = affine.load %arg5[0, 1] : memref<?x4xf64>
    %48 = arith.addf %47, %46 : f64
    affine.store %48, %arg5[0, 1] : memref<?x4xf64>
    %49 = arith.index_cast %arg1 : i32 to index
    %50 = affine.load %arg4[symbol(%49)] : memref<?xf64>
    %51 = arith.mulf %22, %32 : f64
    %52 = arith.mulf %50, %51 : f64
    %53 = affine.load %arg5[0, 2] : memref<?x4xf64>
    %54 = arith.addf %53, %52 : f64
    affine.store %54, %arg5[0, 2] : memref<?x4xf64>
    %55 = arith.index_cast %arg1 : i32 to index
    %56 = affine.load %arg4[symbol(%55)] : memref<?xf64>
    %57 = arith.mulf %22, %37 : f64
    %58 = arith.mulf %56, %57 : f64
    %59 = affine.load %arg5[0, 3] : memref<?x4xf64>
    %60 = arith.addf %59, %58 : f64
    affine.store %60, %arg5[0, 3] : memref<?x4xf64>
    return %arg6 : i32
  }
  func.func @pair_interaction(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>, %arg6: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c45_i32 = arith.constant 45 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sge, %arg6, %c45_i32 : i32
    %1 = arith.select %0, %c1_i32, %c0_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = scf.index_switch %2 -> i32 
    case 0 {
      %4 = func.call @__internal_pair_interaction(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
      scf.yield %4 : i32
    }
    case 1 {
      %4 = func.call @__internal_pair_interaction(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
      scf.yield %4 : i32
    }
    default {
      %4 = func.call @__internal_pair_interaction(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
      scf.yield %4 : i32
    }
    return %3 : i32
  }
  func.func @approx_pair_interaction(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>, %arg6: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = affine.load %arg3[symbol(%0), 0] : memref<?x4xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = affine.load %arg3[symbol(%2), 0] : memref<?x4xf64>
    %4 = arith.addf %1, %3 : f64
    %5 = affine.load %arg3[symbol(%0), 1] : memref<?x4xf64>
    %6 = affine.load %arg3[symbol(%2), 1] : memref<?x4xf64>
    %7 = arith.mulf %5, %6 : f64
    %8 = affine.load %arg3[symbol(%0), 2] : memref<?x4xf64>
    %9 = affine.load %arg3[symbol(%2), 2] : memref<?x4xf64>
    %10 = arith.mulf %8, %9 : f64
    %11 = arith.addf %7, %10 : f64
    %12 = affine.load %arg3[symbol(%0), 3] : memref<?x4xf64>
    %13 = affine.load %arg3[symbol(%2), 3] : memref<?x4xf64>
    %14 = arith.mulf %12, %13 : f64
    %15 = arith.addf %11, %14 : f64
    %16 = arith.subf %4, %15 : f64
    %17 = arith.cmpf olt, %16, %cst_1 : f64
    %18 = arith.select %17, %cst_1, %16 : f64
    %19 = arith.mulf %arg2, %18 : f64
    %20 = arith.addf %19, %cst_0 : f64
    %21 = arith.divf %cst_0, %20 : f64
    %22 = arith.mulf %21, %cst : f64
    %23 = arith.index_cast %arg0 : i32 to index
    %24 = affine.load %arg3[symbol(%23), 1] : memref<?x4xf64>
    %25 = arith.index_cast %arg1 : i32 to index
    %26 = affine.load %arg3[symbol(%25), 1] : memref<?x4xf64>
    %27 = arith.subf %24, %26 : f64
    %28 = arith.index_cast %arg0 : i32 to index
    %29 = affine.load %arg3[symbol(%28), 2] : memref<?x4xf64>
    %30 = arith.index_cast %arg1 : i32 to index
    %31 = affine.load %arg3[symbol(%30), 2] : memref<?x4xf64>
    %32 = arith.subf %29, %31 : f64
    %33 = arith.index_cast %arg0 : i32 to index
    %34 = affine.load %arg3[symbol(%33), 3] : memref<?x4xf64>
    %35 = arith.index_cast %arg1 : i32 to index
    %36 = affine.load %arg3[symbol(%35), 3] : memref<?x4xf64>
    %37 = arith.subf %34, %36 : f64
    %38 = arith.index_cast %arg1 : i32 to index
    %39 = affine.load %arg4[symbol(%38)] : memref<?xf64>
    %40 = arith.mulf %39, %21 : f64
    %41 = affine.load %arg5[0, 0] : memref<?x4xf64>
    %42 = arith.addf %41, %40 : f64
    affine.store %42, %arg5[0, 0] : memref<?x4xf64>
    %43 = arith.index_cast %arg1 : i32 to index
    %44 = affine.load %arg4[symbol(%43)] : memref<?xf64>
    %45 = arith.mulf %22, %27 : f64
    %46 = arith.mulf %44, %45 : f64
    %47 = affine.load %arg5[0, 1] : memref<?x4xf64>
    %48 = arith.addf %47, %46 : f64
    affine.store %48, %arg5[0, 1] : memref<?x4xf64>
    %49 = arith.index_cast %arg1 : i32 to index
    %50 = affine.load %arg4[symbol(%49)] : memref<?xf64>
    %51 = arith.mulf %22, %32 : f64
    %52 = arith.mulf %50, %51 : f64
    %53 = affine.load %arg5[0, 2] : memref<?x4xf64>
    %54 = arith.addf %53, %52 : f64
    affine.store %54, %arg5[0, 2] : memref<?x4xf64>
    %55 = arith.index_cast %arg1 : i32 to index
    %56 = affine.load %arg4[symbol(%55)] : memref<?xf64>
    %57 = arith.mulf %22, %37 : f64
    %58 = arith.mulf %56, %57 : f64
    %59 = affine.load %arg5[0, 3] : memref<?x4xf64>
    %60 = arith.addf %59, %58 : f64
    affine.store %60, %arg5[0, 3] : memref<?x4xf64>
    return %arg6 : i32
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %true = arith.constant true
    %false = arith.constant false
    %c656_i64 = arith.constant 656 : i64
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c656 = arith.constant 656 : index
    %c24 = arith.constant 24 : index
    %c8_i64 = arith.constant 8 : i64
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 1.000000e+06 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+01 : f64
    %c10_i32 = arith.constant 10 : i32
    %c0_i64 = arith.constant 0 : i64
    %c2_i32 = arith.constant 2 : i32
    %c128_i64 = arith.constant 128 : i64
    %cst_3 = arith.constant 5.000000e-01 : f64
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>
    %alloca_4 = memref.alloca() : memref<f64>
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr) -> memref<?x4xf64>
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = "polygeist.pointer2memref"(%5) : (!llvm.ptr) -> memref<?xf64>
    %7 = llvm.mlir.zero : !llvm.ptr
    %8 = "polygeist.pointer2memref"(%7) : (!llvm.ptr) -> memref<?x4xf64>
    %9 = llvm.mlir.addressof @str0 : !llvm.ptr
    %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
    %11 = llvm.call @printf(%10, %c128_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %12 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
    %13 = llvm.getelementptr %12[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
    llvm.store %c1_i32, %13 : i32, !llvm.ptr
    %14 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %15:6 = scf.if %14 -> (i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) {
      %17 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %18 = llvm.mlir.addressof @str1 : !llvm.ptr
      %19 = "polygeist.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi8>
      %20 = func.call @strcmp(%17, %19) : (memref<?xi8>, memref<?xi8>) -> i32
      %21 = arith.cmpi eq, %20, %c0_i32 : i32
      %22 = scf.if %21 -> (i1) {
        %24 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %25 = func.call @isInteger(%24) : (memref<?xi8>) -> i32
        %26 = arith.cmpi ne, %25, %c0_i32 : i32
        scf.yield %26 : i1
      } else {
        scf.yield %false : i1
      }
      %23:6 = scf.if %22 -> (i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) {
        %24 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %25 = llvm.getelementptr %24[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %26 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %27 = func.call @atoi(%26) : (memref<?xi8>) -> i32
        llvm.store %27, %25 : i32, !llvm.ptr
        %28 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %29 = llvm.getelementptr %28[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %30 = llvm.load %29 : !llvm.ptr -> i32
        %31 = arith.cmpi sle, %30, %c0_i32 : i32
        %32 = arith.cmpi sgt, %30, %c0_i32 : i32
        %33 = arith.select %31, %c1_i32, %0 : i32
        scf.if %31 {
          %35 = llvm.mlir.addressof @str2 : !llvm.ptr
          %36 = llvm.getelementptr %35[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
          %37 = llvm.call @printf(%36) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %34:6 = scf.if %32 -> (i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) {
          %35 = llvm.mlir.addressof @str5 : !llvm.ptr
          %36 = llvm.getelementptr %35[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<29 x i8>
          %37 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %38 = llvm.getelementptr %37[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %39 = llvm.load %38 : !llvm.ptr -> i32
          %40 = llvm.call @printf(%36, %39) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
          affine.store %cst_3, %alloca_4[] : memref<f64>
          %41 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %42 = llvm.getelementptr %41[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %43 = llvm.getelementptr %41[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %44 = llvm.load %43 : !llvm.ptr -> i32
          %45 = arith.muli %44, %44 : i32
          %46 = arith.muli %45, %44 : i32
          %47 = arith.extsi %46 : i32 to i64
          llvm.store %47, %42 : i64, !llvm.ptr
          %48 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %49 = llvm.getelementptr %48[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %50 = llvm.getelementptr %48[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %51 = llvm.load %50 : !llvm.ptr -> i64
          %52 = arith.muli %51, %c128_i64 : i64
          llvm.store %52, %49 : i64, !llvm.ptr
          %53 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %54 = llvm.getelementptr %53[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %55 = llvm.load %54 : !llvm.ptr -> i64
          %56 = arith.muli %55, %c656_i64 : i64
          %57 = arith.index_cast %56 : i64 to index
          %58 = arith.divui %57, %c656 : index
          %alloc = memref.alloc(%58) : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
          %59 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %60 = llvm.getelementptr %59[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %61 = llvm.load %60 : !llvm.ptr -> i64
          %62 = arith.muli %61, %c32_i64 : i64
          %63 = arith.index_cast %62 : i64 to index
          %64 = arith.divui %63, %c32 : index
          %alloc_5 = memref.alloc(%64) : memref<?x4xf64>
          %65 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %66 = llvm.getelementptr %65[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %67 = llvm.load %66 : !llvm.ptr -> i64
          %68 = arith.muli %67, %c8_i64 : i64
          %69 = arith.index_cast %68 : i64 to index
          %70 = arith.divui %69, %c8 : index
          %alloc_6 = memref.alloc(%70) : memref<?xf64>
          %71 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %72 = llvm.getelementptr %71[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %73 = llvm.load %72 : !llvm.ptr -> i64
          %74 = arith.muli %73, %c32_i64 : i64
          %75 = arith.index_cast %74 : i64 to index
          %76 = arith.divui %75, %c32 : index
          %alloc_7 = memref.alloc(%76) : memref<?x4xf64>
          %77 = "polygeist.memref2pointer"(%alloc) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
          %78 = llvm.mlir.zero : !llvm.ptr
          %79 = llvm.icmp "eq" %77, %78 : !llvm.ptr
          %80 = scf.if %79 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_5) : (memref<?x4xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %81 = scf.if %80 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_6) : (memref<?xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %82 = scf.if %81 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_7) : (memref<?x4xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %83 = arith.xori %82, %true : i1
          %84 = arith.select %82, %c1_i32, %33 : i32
          scf.if %82 {
            %85 = llvm.mlir.addressof @str6 : !llvm.ptr
            %86 = llvm.getelementptr %85[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12 x i8>
            %87 = llvm.call @printf(%86) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
            memref.dealloc %alloc_5 : memref<?x4xf64>
            memref.dealloc %alloc_6 : memref<?xf64>
            memref.dealloc %alloc_7 : memref<?x4xf64>
            memref.dealloc %alloc : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
          }
          scf.yield %83, %84, %alloc_7, %alloc_6, %alloc_5, %alloc : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
        } else {
          scf.yield %false, %c1_i32, %8, %6, %4, %2 : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
        }
        scf.yield %34#0, %34#1, %34#2, %34#3, %34#4, %34#5 : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
      } else {
        %24 = llvm.mlir.addressof @str3 : !llvm.ptr
        %25 = llvm.getelementptr %24[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
        %26 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
        %27 = "polygeist.memref2pointer"(%26) : (memref<?xi8>) -> !llvm.ptr
        %28 = llvm.call @printf(%25, %27) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        scf.yield %false, %c1_i32, %8, %6, %4, %2 : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
      }
      scf.yield %23#0, %23#1, %23#2, %23#3, %23#4, %23#5 : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
    } else {
      %17 = llvm.mlir.addressof @str4 : !llvm.ptr
      %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<29 x i8>
      %19 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %20 = "polygeist.memref2pointer"(%19) : (memref<?xi8>) -> !llvm.ptr
      %21 = llvm.call @printf(%18, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      scf.yield %false, %c1_i32, %8, %6, %4, %2 : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
    }
    %16 = arith.select %15#0, %c0_i32, %15#1 : i32
    scf.if %15#0 {
      %17 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %18 = llvm.getelementptr %17[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %19 = "polygeist.memref2pointer"(%15#5) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
      %20:2 = scf.while (%arg2 = %c0_i32, %arg3 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %46 = llvm.load %18 : !llvm.ptr -> i32
        %47 = arith.cmpi slt, %arg2, %46 : i32
        scf.condition(%47) %arg2, %arg3 : i32, i32
      } do {
      ^bb0(%arg2: i32, %arg3: i32):
        %46:2 = scf.while (%arg4 = %c0_i32, %arg5 = %arg3) : (i32, i32) -> (i32, i32) {
          %48 = llvm.load %18 : !llvm.ptr -> i32
          %49 = arith.cmpi slt, %arg4, %48 : i32
          scf.condition(%49) %arg5, %arg4 : i32, i32
        } do {
        ^bb0(%arg4: i32, %arg5: i32):
          %48:2 = scf.while (%arg6 = %c0_i32, %arg7 = %arg4) : (i32, i32) -> (i32, i32) {
            %50 = llvm.load %18 : !llvm.ptr -> i32
            %51 = arith.cmpi slt, %arg6, %50 : i32
            scf.condition(%51) %arg7, %arg6 : i32, i32
          } do {
          ^bb0(%arg6: i32, %arg7: i32):
            %50 = arith.index_cast %arg6 : i32 to index
            %51 = arith.muli %50, %c656 : index
            %52 = arith.index_cast %51 : index to i64
            %53 = llvm.getelementptr %19[%52] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %54 = llvm.getelementptr %53[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            llvm.store %arg6, %54 : i32, !llvm.ptr
            %55 = llvm.getelementptr %53[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            %56 = arith.muli %arg6, %c128_i32 : i32
            %57 = arith.extsi %56 : i32 to i64
            llvm.store %57, %55 : i64, !llvm.ptr
            %58 = llvm.getelementptr %53[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            llvm.store %c0_i32, %58 : i32, !llvm.ptr
            scf.for %arg8 = %c-1 to %c2 step %c1 {
              %61 = arith.index_cast %arg8 : index to i32
              scf.for %arg9 = %c-1 to %c2 step %c1 {
                %62 = arith.index_cast %arg9 : index to i32
                scf.for %arg10 = %c-1 to %c2 step %c1 {
                  %63 = arith.index_cast %arg10 : index to i32
                  %64 = arith.cmpi eq, %61, %c0_i32 : i32
                  %65 = arith.cmpi eq, %62, %c0_i32 : i32
                  %66 = arith.andi %64, %65 : i1
                  %67 = scf.if %66 -> (i1) {
                    %74 = arith.cmpi eq, %63, %c0_i32 : i32
                    scf.yield %74 : i1
                  } else {
                    scf.yield %false : i1
                  }
                  %68 = arith.xori %67, %true : i1
                  %69 = arith.addi %arg2, %61 : i32
                  %70 = arith.addi %arg5, %62 : i32
                  %71 = arith.addi %arg7, %63 : i32
                  %72 = arith.cmpi sge, %69, %c0_i32 : i32
                  %73 = arith.andi %68, %72 : i1
                  scf.if %73 {
                    %74 = llvm.load %18 : !llvm.ptr -> i32
                    %75 = arith.cmpi slt, %69, %74 : i32
                    %76 = arith.cmpi sge, %70, %c0_i32 : i32
                    %77 = arith.andi %75, %76 : i1
                    scf.if %77 {
                      %78 = llvm.load %18 : !llvm.ptr -> i32
                      %79 = arith.cmpi slt, %70, %78 : i32
                      %80 = arith.cmpi sge, %71, %c0_i32 : i32
                      %81 = arith.andi %79, %80 : i1
                      scf.if %81 {
                        %82 = llvm.load %18 : !llvm.ptr -> i32
                        %83 = arith.cmpi slt, %71, %82 : i32
                        scf.if %83 {
                          %84 = llvm.load %58 : !llvm.ptr -> i32
                          %85 = arith.addi %84, %c1_i32 : i32
                          llvm.store %85, %58 : i32, !llvm.ptr
                          %86 = llvm.load %18 : !llvm.ptr -> i32
                          %87 = arith.muli %69, %86 : i32
                          %88 = arith.muli %87, %86 : i32
                          %89 = arith.muli %70, %86 : i32
                          %90 = arith.addi %88, %89 : i32
                          %91 = arith.addi %90, %71 : i32
                          %92 = llvm.getelementptr %53[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
                          %93 = arith.index_cast %84 : i32 to index
                          %94 = arith.muli %93, %c24 : index
                          %95 = arith.index_cast %94 : index to i64
                          %96 = llvm.getelementptr %92[%95] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                          %97 = llvm.getelementptr %96[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
                          llvm.store %91, %97 : i32, !llvm.ptr
                          %98 = llvm.getelementptr %96[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
                          %99 = arith.muli %91, %c128_i32 : i32
                          %100 = arith.extsi %99 : i32 to i64
                          llvm.store %100, %98 : i64, !llvm.ptr
                        }
                      }
                    }
                  }
                }
              }
            }
            %59 = arith.addi %arg6, %c1_i32 : i32
            %60 = arith.addi %arg7, %c1_i32 : i32
            scf.yield %60, %59 : i32, i32
          }
          %49 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %49, %48#0 : i32, i32
        }
        %47 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %47, %46#0 : i32, i32
      }
      func.call @srand(%c2_i32) : (i32) -> ()
      %21 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %22 = llvm.getelementptr %21[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %23 = scf.while (%arg2 = %c0_i64) : (i64) -> i64 {
        %46 = llvm.load %22 : !llvm.ptr -> i64
        %47 = arith.cmpi slt, %arg2, %46 : i64
        scf.condition(%47) %arg2 : i64
      } do {
      ^bb0(%arg2: i64):
        %46 = arith.index_cast %arg2 : i64 to index
        %47 = func.call @rand() : () -> i32
        %48 = arith.remsi %47, %c10_i32 : i32
        %49 = arith.addi %48, %c1_i32 : i32
        %50 = arith.sitofp %49 : i32 to f64
        %51 = arith.divf %50, %cst_2 : f64
        memref.store %51, %15#4[%46, %c0] : memref<?x4xf64>
        %52 = func.call @rand() : () -> i32
        %53 = arith.remsi %52, %c10_i32 : i32
        %54 = arith.addi %53, %c1_i32 : i32
        %55 = arith.sitofp %54 : i32 to f64
        %56 = arith.divf %55, %cst_2 : f64
        memref.store %56, %15#4[%46, %c1] : memref<?x4xf64>
        %57 = func.call @rand() : () -> i32
        %58 = arith.remsi %57, %c10_i32 : i32
        %59 = arith.addi %58, %c1_i32 : i32
        %60 = arith.sitofp %59 : i32 to f64
        %61 = arith.divf %60, %cst_2 : f64
        memref.store %61, %15#4[%46, %c2] : memref<?x4xf64>
        %62 = func.call @rand() : () -> i32
        %63 = arith.remsi %62, %c10_i32 : i32
        %64 = arith.addi %63, %c1_i32 : i32
        %65 = arith.sitofp %64 : i32 to f64
        %66 = arith.divf %65, %cst_2 : f64
        memref.store %66, %15#4[%46, %c3] : memref<?x4xf64>
        %67 = func.call @rand() : () -> i32
        %68 = arith.remsi %67, %c10_i32 : i32
        %69 = arith.addi %68, %c1_i32 : i32
        %70 = arith.sitofp %69 : i32 to f64
        %71 = arith.divf %70, %cst_2 : f64
        memref.store %71, %15#3[%46] : memref<?xf64>
        memref.store %cst_1, %15#2[%46, %c3] : memref<?x4xf64>
        %72 = memref.load %15#2[%46, %c3] : memref<?x4xf64>
        memref.store %72, %15#2[%46, %c2] : memref<?x4xf64>
        %73 = memref.load %15#2[%46, %c2] : memref<?x4xf64>
        memref.store %73, %15#2[%46, %c1] : memref<?x4xf64>
        %74 = memref.load %15#2[%46, %c1] : memref<?x4xf64>
        memref.store %74, %15#2[%46, %c0] : memref<?x4xf64>
        %75 = arith.addi %arg2, %c1_i64 : i64
        scf.yield %75 : i64
      }
      %24 = func.call @get_time() : () -> i64
      %25 = affine.load %alloca_4[] : memref<f64>
      %26 = arith.mulf %25, %cst_0 : f64
      %27 = arith.mulf %26, %25 : f64
      %28 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %29 = llvm.getelementptr %28[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %30 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %46 = arith.extsi %arg2 : i32 to i64
        %47 = llvm.load %29 : !llvm.ptr -> i64
        %48 = arith.cmpi slt, %46, %47 : i64
        scf.condition(%48) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        func.call @process_home_box(%arg2, %27, %15#5, %15#4, %15#3, %15#2) : (i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> ()
        %46 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %46 : i32
      }
      %31 = func.call @get_time() : () -> i64
      %32 = llvm.mlir.addressof @str7 : !llvm.ptr
      %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
      %34 = arith.subi %31, %24 : i64
      %35 = arith.sitofp %34 : i64 to f64
      %36 = arith.divf %35, %cst : f64
      %37 = llvm.call @printf(%33, %36) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      %38 = llvm.mlir.addressof @str8 : !llvm.ptr
      %39 = llvm.mlir.addressof @str9 : !llvm.ptr
      %40 = "polygeist.pointer2memref"(%38) : (!llvm.ptr) -> memref<?xi8>
      %41 = "polygeist.pointer2memref"(%39) : (!llvm.ptr) -> memref<?xi8>
      %42 = func.call @fopen(%40, %41) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %43 = "polygeist.memref2pointer"(%42) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %44 = llvm.mlir.zero : !llvm.ptr
      %45 = llvm.icmp "ne" %43, %44 : !llvm.ptr
      scf.if %45 {
        %46 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %47 = llvm.getelementptr %46[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %48 = "polygeist.pointer2memref"(%47) : (!llvm.ptr) -> memref<?xi8>
        %49 = func.call @fwrite(%48, %c8_i64, %c1_i64, %42) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %50 = arith.cmpi ne, %49, %c1_i64 : i64
        scf.if %50 {
          %65 = llvm.mlir.addressof @str10 : !llvm.ptr
          %66 = llvm.getelementptr %65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
          %67 = llvm.call @printf(%66) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %51 = "polygeist.memref2pointer"(%15#2) : (memref<?x4xf64>) -> !llvm.ptr
        %52 = "polygeist.pointer2memref"(%51) : (!llvm.ptr) -> memref<?xi8>
        %53 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %54 = llvm.getelementptr %53[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %55 = llvm.load %54 : !llvm.ptr -> i64
        %56 = func.call @fwrite(%52, %c32_i64, %55, %42) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %57 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %58 = llvm.getelementptr %57[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %59 = llvm.load %58 : !llvm.ptr -> i64
        %60 = arith.cmpi ne, %56, %59 : i64
        scf.if %60 {
          %65 = llvm.mlir.addressof @str11 : !llvm.ptr
          %66 = llvm.getelementptr %65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
          %67 = llvm.call @printf(%66) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %61 = func.call @fclose(%42) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
        %62 = llvm.mlir.addressof @str12 : !llvm.ptr
        %63 = llvm.getelementptr %62[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31 x i8>
        %64 = llvm.call @printf(%63) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      } else {
        %46 = llvm.mlir.addressof @str13 : !llvm.ptr
        %47 = llvm.getelementptr %46[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
        %48 = llvm.call @printf(%47) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      }
      memref.dealloc %15#4 : memref<?x4xf64>
      memref.dealloc %15#3 : memref<?xf64>
      memref.dealloc %15#2 : memref<?x4xf64>
      memref.dealloc %15#5 : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
    }
    return %16 : i32
  }
  func.func private @strcmp(memref<?xi8>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @isInteger(%arg0: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c57_i32 = arith.constant 57 : i32
    %c48_i32 = arith.constant 48 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = affine.load %arg0[0] : memref<?xi8>
      %9 = arith.cmpi eq, %8, %c0_i8 : i8
      scf.yield %9 : i1
    }
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6:2 = scf.if %4 -> (i1, i32) {
      scf.yield %false, %5 : i1, i32
    } else {
      %8:4 = scf.while (%arg1 = %arg0, %arg2 = %true, %arg3 = %5, %arg4 = %true) : (memref<?xi8>, i1, i32, i1) -> (i1, i32, i8, memref<?xi8>) {
        %9 = affine.load %arg1[0] : memref<?xi8>
        %10 = arith.cmpi ne, %9, %c0_i8 : i8
        %11 = arith.andi %10, %arg4 : i1
        scf.condition(%11) %arg2, %arg3, %9, %arg1 : i1, i32, i8, memref<?xi8>
      } do {
      ^bb0(%arg1: i1, %arg2: i32, %arg3: i8, %arg4: memref<?xi8>):
        %9 = arith.extsi %arg3 : i8 to i32
        %10 = arith.cmpi slt, %9, %c48_i32 : i32
        %11 = scf.if %10 -> (i1) {
          scf.yield %true : i1
        } else {
          %17 = arith.extsi %arg3 : i8 to i32
          %18 = arith.cmpi sgt, %17, %c57_i32 : i32
          scf.yield %18 : i1
        }
        %12 = arith.xori %11, %true : i1
        %13 = arith.andi %12, %arg1 : i1
        %14 = arith.select %11, %c0_i32, %arg2 : i32
        %15 = arith.xori %11, %true : i1
        %16 = scf.if %11 -> (memref<?xi8>) {
          scf.yield %arg4 : memref<?xi8>
        } else {
          %17 = "polygeist.subindex"(%arg4, %c1) : (memref<?xi8>, index) -> memref<?xi8>
          scf.yield %17 : memref<?xi8>
        }
        scf.yield %16, %13, %14, %15 : memref<?xi8>, i1, i32, i1
      }
      scf.yield %8#0, %8#1 : i1, i32
    }
    %7 = arith.select %6#0, %c1_i32, %6#1 : i32
    return %7 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @srand(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @get_time() -> i64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1000000_i64 = arith.constant 1000000 : i64
    %alloca = memref.alloca() : memref<1x2xi64>
    %cast = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi8>
    %2 = call @gettimeofday(%cast, %1) : (memref<?x2xi64>, memref<?xi8>) -> i32
    %3 = affine.load %alloca[0, 0] : memref<1x2xi64>
    %4 = arith.muli %3, %c1000000_i64 : i64
    %5 = affine.load %alloca[0, 1] : memref<1x2xi64>
    %6 = arith.addi %4, %5 : i64
    return %6 : i64
  }
  func.func private @process_home_box(%arg0: i32, %arg1: f64, %arg2: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c656 = arith.constant 656 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %alloca = memref.alloca() : memref<1x4xf64>
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.muli %0, %c656 : index
    %2 = arith.index_cast %1 : index to i64
    %3 = "polygeist.memref2pointer"(%arg2) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
    %4 = llvm.getelementptr %3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %5 = llvm.getelementptr %4[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %6 = llvm.load %5 : !llvm.ptr -> i64
    %7 = arith.trunci %6 : i64 to i32
    scf.for %arg6 = %c0 to %c128 step %c1 {
      %8 = arith.index_cast %arg6 : index to i32
      %9 = arith.addi %7, %8 : i32
      affine.store %cst, %alloca[0, 0] : memref<1x4xf64>
      affine.store %cst, %alloca[0, 1] : memref<1x4xf64>
      affine.store %cst, %alloca[0, 2] : memref<1x4xf64>
      affine.store %cst, %alloca[0, 3] : memref<1x4xf64>
      %10 = arith.index_cast %9 : i32 to index
      %11 = memref.load %arg4[%10] : memref<?xf64>
      %12 = func.call @state_self_from_qi(%11) : (f64) -> i32
      %13 = llvm.getelementptr %4[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %14 = llvm.load %13 : !llvm.ptr -> i32
      %15 = func.call @state_neigh_from_nn(%14) : (i32) -> i32
      %cast = memref.cast %alloca : memref<1x4xf64> to memref<?x4xf64>
      func.call @self_box_accumulate(%9, %7, %arg1, %arg3, %arg4, %cast, %12) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> ()
      func.call @neighbor_box_accumulate(%9, %arg0, %arg1, %arg2, %arg3, %arg4, %cast, %15) : (i32, i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> ()
      %16 = affine.load %alloca[0, 0] : memref<1x4xf64>
      %17 = memref.load %arg5[%10, %c0] : memref<?x4xf64>
      %18 = arith.addf %17, %16 : f64
      memref.store %18, %arg5[%10, %c0] : memref<?x4xf64>
      %19 = affine.load %alloca[0, 1] : memref<1x4xf64>
      %20 = memref.load %arg5[%10, %c1] : memref<?x4xf64>
      %21 = arith.addf %20, %19 : f64
      memref.store %21, %arg5[%10, %c1] : memref<?x4xf64>
      %22 = affine.load %alloca[0, 2] : memref<1x4xf64>
      %23 = memref.load %arg5[%10, %c2] : memref<?x4xf64>
      %24 = arith.addf %23, %22 : f64
      memref.store %24, %arg5[%10, %c2] : memref<?x4xf64>
      %25 = affine.load %alloca[0, 3] : memref<1x4xf64>
      %26 = memref.load %arg5[%10, %c3] : memref<?x4xf64>
      %27 = arith.addf %26, %25 : f64
      memref.store %27, %arg5[%10, %c3] : memref<?x4xf64>
    }
    return
  }
  func.func private @fopen(memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fwrite(memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fclose(memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @gettimeofday(memref<?x2xi64>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @state_self_from_qi(%arg0: f64) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 1.000000e+02 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %0 = arith.cmpf olt, %arg0, %cst_1 : f64
    %1 = scf.if %0 -> (f64) {
      %7 = arith.negf %arg0 : f64
      scf.yield %7 : f64
    } else {
      scf.yield %arg0 : f64
    }
    %2 = arith.mulf %1, %cst_0 : f64
    %3 = arith.cmpf ogt, %2, %cst_0 : f64
    %4 = arith.select %3, %cst_0, %2 : f64
    %5 = arith.addf %4, %cst : f64
    %6 = arith.fptosi %5 : f64 to i32
    return %6 : i32
  }
  func.func private @state_neigh_from_nn(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c26_i32 = arith.constant 26 : i32
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi slt, %arg0, %c0_i32 : i32
    %1 = arith.select %0, %c0_i32, %arg0 : i32
    %2 = scf.if %0 -> (i1) {
      scf.yield %false : i1
    } else {
      %4 = arith.cmpi sgt, %arg0, %c26_i32 : i32
      scf.yield %4 : i1
    }
    %3 = arith.select %2, %c26_i32, %1 : i32
    return %3 : i32
  }
  func.func private @self_box_accumulate(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c25_i32 = arith.constant 25 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.cmpi sge, %arg6, %c25_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.index_switch %3 
    case 0 {
      scf.for %arg7 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg7 : index to i32
        %5 = arith.addi %arg1, %4 : i32
        %6 = arith.cmpi ne, %arg0, %5 : i32
        scf.if %6 {
          %7 = affine.load %arg3[symbol(%0), 0] : memref<?x4xf64>
          %8 = arith.index_cast %5 : i32 to index
          %9 = memref.load %arg3[%8, %c0] : memref<?x4xf64>
          %10 = arith.addf %7, %9 : f64
          %11 = affine.load %arg3[symbol(%0), 1] : memref<?x4xf64>
          %12 = memref.load %arg3[%8, %c1] : memref<?x4xf64>
          %13 = arith.mulf %11, %12 : f64
          %14 = affine.load %arg3[symbol(%0), 2] : memref<?x4xf64>
          %15 = memref.load %arg3[%8, %c2] : memref<?x4xf64>
          %16 = arith.mulf %14, %15 : f64
          %17 = arith.addf %13, %16 : f64
          %18 = affine.load %arg3[symbol(%0), 3] : memref<?x4xf64>
          %19 = memref.load %arg3[%8, %c3] : memref<?x4xf64>
          %20 = arith.mulf %18, %19 : f64
          %21 = arith.addf %17, %20 : f64
          %22 = arith.subf %10, %21 : f64
          %23 = arith.cmpf olt, %22, %cst : f64
          %24 = arith.select %23, %cst, %22 : f64
          %25 = arith.mulf %arg2, %24 : f64
          %26 = func.call @state_pair_from_u2(%25) : (f64) -> i32
          %27 = func.call @pair_interaction(%arg0, %5, %arg2, %arg3, %arg4, %arg5, %26) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
        }
      }
      scf.yield
    }
    case 1 {
      scf.for %arg7 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg7 : index to i32
        %5 = arith.addi %arg1, %4 : i32
        %6 = arith.cmpi ne, %arg0, %5 : i32
        scf.if %6 {
          %7 = affine.load %arg3[symbol(%0), 0] : memref<?x4xf64>
          %8 = arith.index_cast %5 : i32 to index
          %9 = memref.load %arg3[%8, %c0] : memref<?x4xf64>
          %10 = arith.addf %7, %9 : f64
          %11 = affine.load %arg3[symbol(%0), 1] : memref<?x4xf64>
          %12 = memref.load %arg3[%8, %c1] : memref<?x4xf64>
          %13 = arith.mulf %11, %12 : f64
          %14 = affine.load %arg3[symbol(%0), 2] : memref<?x4xf64>
          %15 = memref.load %arg3[%8, %c2] : memref<?x4xf64>
          %16 = arith.mulf %14, %15 : f64
          %17 = arith.addf %13, %16 : f64
          %18 = affine.load %arg3[symbol(%0), 3] : memref<?x4xf64>
          %19 = memref.load %arg3[%8, %c3] : memref<?x4xf64>
          %20 = arith.mulf %18, %19 : f64
          %21 = arith.addf %17, %20 : f64
          %22 = arith.subf %10, %21 : f64
          %23 = arith.cmpf olt, %22, %cst : f64
          %24 = arith.select %23, %cst, %22 : f64
          %25 = arith.mulf %arg2, %24 : f64
          %26 = func.call @state_pair_from_u2(%25) : (f64) -> i32
          %27 = func.call @pair_interaction(%arg0, %5, %arg2, %arg3, %arg4, %arg5, %26) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
        }
      }
      scf.yield
    }
    default {
      scf.for %arg7 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg7 : index to i32
        %5 = arith.addi %arg1, %4 : i32
        %6 = arith.cmpi ne, %arg0, %5 : i32
        scf.if %6 {
          %7 = affine.load %arg3[symbol(%0), 0] : memref<?x4xf64>
          %8 = arith.index_cast %5 : i32 to index
          %9 = memref.load %arg3[%8, %c0] : memref<?x4xf64>
          %10 = arith.addf %7, %9 : f64
          %11 = affine.load %arg3[symbol(%0), 1] : memref<?x4xf64>
          %12 = memref.load %arg3[%8, %c1] : memref<?x4xf64>
          %13 = arith.mulf %11, %12 : f64
          %14 = affine.load %arg3[symbol(%0), 2] : memref<?x4xf64>
          %15 = memref.load %arg3[%8, %c2] : memref<?x4xf64>
          %16 = arith.mulf %14, %15 : f64
          %17 = arith.addf %13, %16 : f64
          %18 = affine.load %arg3[symbol(%0), 3] : memref<?x4xf64>
          %19 = memref.load %arg3[%8, %c3] : memref<?x4xf64>
          %20 = arith.mulf %18, %19 : f64
          %21 = arith.addf %17, %20 : f64
          %22 = arith.subf %10, %21 : f64
          %23 = arith.cmpf olt, %22, %cst : f64
          %24 = arith.select %23, %cst, %22 : f64
          %25 = arith.mulf %arg2, %24 : f64
          %26 = func.call @state_pair_from_u2(%25) : (f64) -> i32
          %27 = func.call @pair_interaction(%arg0, %5, %arg2, %arg3, %arg4, %arg5, %26) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
        }
      }
    }
    return
  }
  func.func private @neighbor_box_accumulate(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg4: memref<?x4xf64>, %arg5: memref<?xf64>, %arg6: memref<?x4xf64>, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c24 = arith.constant 24 : index
    %c656 = arith.constant 656 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.cmpi sge, %arg7, %c5_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.index_switch %3 
    case 0 {
      %4 = arith.index_cast %arg1 : i32 to index
      %5 = arith.muli %4, %c656 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
      %8 = llvm.getelementptr %7[%6] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %9 = llvm.getelementptr %8[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %10 = llvm.getelementptr %8[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %11 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
        %12 = llvm.load %9 : !llvm.ptr -> i32
        %13 = arith.cmpi slt, %arg8, %12 : i32
        scf.condition(%13) %arg8 : i32
      } do {
      ^bb0(%arg8: i32):
        %12 = arith.index_cast %arg8 : i32 to index
        %13 = arith.muli %12, %c24 : index
        %14 = arith.index_cast %13 : index to i64
        %15 = llvm.getelementptr %10[%14] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %16 = llvm.getelementptr %15[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
        %17 = llvm.load %16 : !llvm.ptr -> i32
        %18 = arith.index_cast %17 : i32 to index
        %19 = arith.muli %18, %c656 : index
        %20 = arith.index_cast %19 : index to i64
        %21 = llvm.getelementptr %7[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %22 = llvm.getelementptr %21[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
        %23 = llvm.load %22 : !llvm.ptr -> i64
        %24 = arith.trunci %23 : i64 to i32
        scf.for %arg9 = %c0 to %c128 step %c1 {
          %26 = arith.index_cast %arg9 : index to i32
          %27 = arith.addi %24, %26 : i32
          %28 = affine.load %arg4[symbol(%0), 0] : memref<?x4xf64>
          %29 = arith.index_cast %27 : i32 to index
          %30 = memref.load %arg4[%29, %c0] : memref<?x4xf64>
          %31 = arith.addf %28, %30 : f64
          %32 = affine.load %arg4[symbol(%0), 1] : memref<?x4xf64>
          %33 = memref.load %arg4[%29, %c1] : memref<?x4xf64>
          %34 = arith.mulf %32, %33 : f64
          %35 = affine.load %arg4[symbol(%0), 2] : memref<?x4xf64>
          %36 = memref.load %arg4[%29, %c2] : memref<?x4xf64>
          %37 = arith.mulf %35, %36 : f64
          %38 = arith.addf %34, %37 : f64
          %39 = affine.load %arg4[symbol(%0), 3] : memref<?x4xf64>
          %40 = memref.load %arg4[%29, %c3] : memref<?x4xf64>
          %41 = arith.mulf %39, %40 : f64
          %42 = arith.addf %38, %41 : f64
          %43 = arith.subf %31, %42 : f64
          %44 = arith.cmpf olt, %43, %cst : f64
          %45 = arith.select %44, %cst, %43 : f64
          %46 = arith.mulf %arg2, %45 : f64
          %47 = func.call @state_pair_from_u2(%46) : (f64) -> i32
          %48 = func.call @pair_interaction(%arg0, %27, %arg2, %arg4, %arg5, %arg6, %47) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
        }
        %25 = arith.addi %arg8, %c1_i32 : i32
        scf.yield %25 : i32
      }
      scf.yield
    }
    case 1 {
      %4 = arith.index_cast %arg1 : i32 to index
      %5 = arith.muli %4, %c656 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
      %8 = llvm.getelementptr %7[%6] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %9 = llvm.getelementptr %8[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %10 = llvm.getelementptr %8[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %11 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
        %12 = llvm.load %9 : !llvm.ptr -> i32
        %13 = arith.cmpi slt, %arg8, %12 : i32
        scf.condition(%13) %arg8 : i32
      } do {
      ^bb0(%arg8: i32):
        %12 = arith.index_cast %arg8 : i32 to index
        %13 = arith.muli %12, %c24 : index
        %14 = arith.index_cast %13 : index to i64
        %15 = llvm.getelementptr %10[%14] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %16 = llvm.getelementptr %15[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
        %17 = llvm.load %16 : !llvm.ptr -> i32
        %18 = arith.index_cast %17 : i32 to index
        %19 = arith.muli %18, %c656 : index
        %20 = arith.index_cast %19 : index to i64
        %21 = llvm.getelementptr %7[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %22 = llvm.getelementptr %21[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
        %23 = llvm.load %22 : !llvm.ptr -> i64
        %24 = arith.trunci %23 : i64 to i32
        scf.for %arg9 = %c0 to %c128 step %c1 {
          %26 = arith.index_cast %arg9 : index to i32
          %27 = arith.addi %24, %26 : i32
          %28 = affine.load %arg4[symbol(%0), 0] : memref<?x4xf64>
          %29 = arith.index_cast %27 : i32 to index
          %30 = memref.load %arg4[%29, %c0] : memref<?x4xf64>
          %31 = arith.addf %28, %30 : f64
          %32 = affine.load %arg4[symbol(%0), 1] : memref<?x4xf64>
          %33 = memref.load %arg4[%29, %c1] : memref<?x4xf64>
          %34 = arith.mulf %32, %33 : f64
          %35 = affine.load %arg4[symbol(%0), 2] : memref<?x4xf64>
          %36 = memref.load %arg4[%29, %c2] : memref<?x4xf64>
          %37 = arith.mulf %35, %36 : f64
          %38 = arith.addf %34, %37 : f64
          %39 = affine.load %arg4[symbol(%0), 3] : memref<?x4xf64>
          %40 = memref.load %arg4[%29, %c3] : memref<?x4xf64>
          %41 = arith.mulf %39, %40 : f64
          %42 = arith.addf %38, %41 : f64
          %43 = arith.subf %31, %42 : f64
          %44 = arith.cmpf olt, %43, %cst : f64
          %45 = arith.select %44, %cst, %43 : f64
          %46 = arith.mulf %arg2, %45 : f64
          %47 = func.call @state_pair_from_u2(%46) : (f64) -> i32
          %48 = func.call @pair_interaction(%arg0, %27, %arg2, %arg4, %arg5, %arg6, %47) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
        }
        %25 = arith.addi %arg8, %c1_i32 : i32
        scf.yield %25 : i32
      }
      scf.yield
    }
    default {
      %4 = arith.index_cast %arg1 : i32 to index
      %5 = arith.muli %4, %c656 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
      %8 = llvm.getelementptr %7[%6] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %9 = llvm.getelementptr %8[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %10 = llvm.getelementptr %8[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %11 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
        %12 = llvm.load %9 : !llvm.ptr -> i32
        %13 = arith.cmpi slt, %arg8, %12 : i32
        scf.condition(%13) %arg8 : i32
      } do {
      ^bb0(%arg8: i32):
        %12 = arith.index_cast %arg8 : i32 to index
        %13 = arith.muli %12, %c24 : index
        %14 = arith.index_cast %13 : index to i64
        %15 = llvm.getelementptr %10[%14] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %16 = llvm.getelementptr %15[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
        %17 = llvm.load %16 : !llvm.ptr -> i32
        %18 = arith.index_cast %17 : i32 to index
        %19 = arith.muli %18, %c656 : index
        %20 = arith.index_cast %19 : index to i64
        %21 = llvm.getelementptr %7[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %22 = llvm.getelementptr %21[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
        %23 = llvm.load %22 : !llvm.ptr -> i64
        %24 = arith.trunci %23 : i64 to i32
        scf.for %arg9 = %c0 to %c128 step %c1 {
          %26 = arith.index_cast %arg9 : index to i32
          %27 = arith.addi %24, %26 : i32
          %28 = affine.load %arg4[symbol(%0), 0] : memref<?x4xf64>
          %29 = arith.index_cast %27 : i32 to index
          %30 = memref.load %arg4[%29, %c0] : memref<?x4xf64>
          %31 = arith.addf %28, %30 : f64
          %32 = affine.load %arg4[symbol(%0), 1] : memref<?x4xf64>
          %33 = memref.load %arg4[%29, %c1] : memref<?x4xf64>
          %34 = arith.mulf %32, %33 : f64
          %35 = affine.load %arg4[symbol(%0), 2] : memref<?x4xf64>
          %36 = memref.load %arg4[%29, %c2] : memref<?x4xf64>
          %37 = arith.mulf %35, %36 : f64
          %38 = arith.addf %34, %37 : f64
          %39 = affine.load %arg4[symbol(%0), 3] : memref<?x4xf64>
          %40 = memref.load %arg4[%29, %c3] : memref<?x4xf64>
          %41 = arith.mulf %39, %40 : f64
          %42 = arith.addf %38, %41 : f64
          %43 = arith.subf %31, %42 : f64
          %44 = arith.cmpf olt, %43, %cst : f64
          %45 = arith.select %44, %cst, %43 : f64
          %46 = arith.mulf %arg2, %45 : f64
          %47 = func.call @state_pair_from_u2(%46) : (f64) -> i32
          %48 = func.call @pair_interaction(%arg0, %27, %arg2, %arg4, %arg5, %arg6, %47) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
        }
        %25 = arith.addi %arg8, %c1_i32 : i32
        scf.yield %25 : i32
      }
    }
    return
  }
  func.func private @state_pair_from_u2(%arg0: f64) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 1.000000e+02 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %0 = arith.cmpf olt, %arg0, %cst_1 : f64
    %1 = arith.select %0, %cst_1, %arg0 : f64
    %2 = arith.mulf %1, %cst_0 : f64
    %3 = arith.addf %2, %cst : f64
    %4 = arith.fptosi %3 : f64 to i32
    return %4 : i32
  }
}

