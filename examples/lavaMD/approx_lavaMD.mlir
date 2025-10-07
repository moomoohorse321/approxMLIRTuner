module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  // B) self-box inner loop perforation (loop_perforate)
  // decisions: 1 = step 1 (exact), 2 = step 2 (perforated)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "self_box_accumulate",
    transform_type = "loop_perforate",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 100>,   // state in [0,100] (= |q_i| * 100)
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 1, 2, 3, 4>,
    thresholds = array<i32: 5>,           // |q_i| >= 0.60 => step=1 else step=2
    decisions = array<i32: 1, 4>
  }> : () -> ()

  // C) neighbor-box perforation (loop_perforate over k/j)
  // decisions: 1 = step 1 (exact), 2 = step 2 (perforated)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "neighbor_box_accumulate",
    transform_type = "loop_perforate",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 10>,    // state = nn in [0,26]
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 1, 2, 3, 4>,
    thresholds = array<i32: 9>,           // nn >= 13 => step=1 else step=2
    decisions = array<i32: 2, 4>
  }> : () -> ()
  llvm.mlir.global internal constant @str28("ERROR: open result.txt\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str27("Results written to result.txt\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str26("ERROR: data write\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str25("ERROR: header write\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str24("wb\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str23("result.txt\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str22("%14zu | %13.6f | %11.6f | %11.6f | %11.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str21("---------------|---------------|-------------|-------------|-------------\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str20("Particle Index | Potential (v) |   Force (x) |   Force (y) |   Force (z)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str19("\0A--- Full Particle Data Dump (fv_cpu) ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str18("      Force Vector (z) | %12.6f | %12.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("      Force Vector (y) | %12.6f | %12.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("      Force Vector (x) | %12.6f | %12.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("Potential Energy (v) | %12.6f | %12.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("------------------|--------------|--------------|--------------\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("        Component |      Average |          Min |          Max\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("\0A--- Result Summary (fv_cpu) ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Particles per Box: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("Number of Boxes: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Total Particles: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("\0A--- Simulation Statistics ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("Total execution time: %.3f ms\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: OOM\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("Configuration: boxes1d = %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("Usage: %s -boxes1d <number>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("ERROR: Usage: %s -boxes1d <number>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("ERROR: -boxes1d > 0\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("-boxes1d\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("WG size of kernel = %d\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @pair_interaction(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
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
    %23 = arith.index_cast %arg1 : i32 to index
    %24 = affine.load %arg4[symbol(%23)] : memref<?xf64>
    %25 = arith.mulf %24, %21 : f64
    %26 = affine.load %arg5[0, 0] : memref<?x4xf64>
    %27 = arith.addf %26, %25 : f64
    affine.store %27, %arg5[0, 0] : memref<?x4xf64>
    %28 = arith.index_cast %arg1 : i32 to index
    %29 = affine.load %arg4[symbol(%28)] : memref<?xf64>
    %30 = arith.index_cast %arg0 : i32 to index
    %31 = affine.load %arg3[symbol(%30), 1] : memref<?x4xf64>
    %32 = affine.load %arg3[symbol(%28), 1] : memref<?x4xf64>
    %33 = arith.subf %31, %32 : f64
    %34 = arith.mulf %22, %33 : f64
    %35 = arith.mulf %29, %34 : f64
    %36 = affine.load %arg5[0, 1] : memref<?x4xf64>
    %37 = arith.addf %36, %35 : f64
    affine.store %37, %arg5[0, 1] : memref<?x4xf64>
    %38 = arith.index_cast %arg1 : i32 to index
    %39 = affine.load %arg4[symbol(%38)] : memref<?xf64>
    %40 = arith.index_cast %arg0 : i32 to index
    %41 = affine.load %arg3[symbol(%40), 2] : memref<?x4xf64>
    %42 = affine.load %arg3[symbol(%38), 2] : memref<?x4xf64>
    %43 = arith.subf %41, %42 : f64
    %44 = arith.mulf %22, %43 : f64
    %45 = arith.mulf %39, %44 : f64
    %46 = affine.load %arg5[0, 2] : memref<?x4xf64>
    %47 = arith.addf %46, %45 : f64
    affine.store %47, %arg5[0, 2] : memref<?x4xf64>
    %48 = arith.index_cast %arg1 : i32 to index
    %49 = affine.load %arg4[symbol(%48)] : memref<?xf64>
    %50 = arith.index_cast %arg0 : i32 to index
    %51 = affine.load %arg3[symbol(%50), 3] : memref<?x4xf64>
    %52 = affine.load %arg3[symbol(%48), 3] : memref<?x4xf64>
    %53 = arith.subf %51, %52 : f64
    %54 = arith.mulf %22, %53 : f64
    %55 = arith.mulf %49, %54 : f64
    %56 = affine.load %arg5[0, 3] : memref<?x4xf64>
    %57 = arith.addf %56, %55 : f64
    affine.store %57, %arg5[0, 3] : memref<?x4xf64>
    return %c0_i32 : i32
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
    %cst = arith.constant 0x42008E8D71C00000 : f64
    %cst_0 = arith.constant 0xC2008E8D71C00000 : f64
    %cst_1 = arith.constant 1.000000e+06 : f64
    %cst_2 = arith.constant 1.000000e+03 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %cst_4 = arith.constant 0.000000e+00 : f64
    %cst_5 = arith.constant 1.000000e+01 : f64
    %c10_i32 = arith.constant 10 : i32
    %c0_i64 = arith.constant 0 : i64
    %c2_i32 = arith.constant 2 : i32
    %c128_i64 = arith.constant 128 : i64
    %cst_6 = arith.constant 5.000000e-01 : f64
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1x2xi64>
    %alloca_7 = memref.alloca() : memref<1x2xi64>
    %alloca_8 = memref.alloca() : memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>
    %alloca_9 = memref.alloca() : memref<f64>
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
    %12 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
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
        %24 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %25 = llvm.getelementptr %24[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %26 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %27 = func.call @atoi(%26) : (memref<?xi8>) -> i32
        llvm.store %27, %25 : i32, !llvm.ptr
        %28 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
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
          %37 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %38 = llvm.getelementptr %37[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %39 = llvm.load %38 : !llvm.ptr -> i32
          %40 = llvm.call @printf(%36, %39) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
          affine.store %cst_6, %alloca_9[] : memref<f64>
          %41 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %42 = llvm.getelementptr %41[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %43 = llvm.getelementptr %41[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %44 = llvm.load %43 : !llvm.ptr -> i32
          %45 = arith.muli %44, %44 : i32
          %46 = arith.muli %45, %44 : i32
          %47 = arith.extsi %46 : i32 to i64
          llvm.store %47, %42 : i64, !llvm.ptr
          %48 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %49 = llvm.getelementptr %48[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %50 = llvm.getelementptr %48[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %51 = llvm.load %50 : !llvm.ptr -> i64
          %52 = arith.muli %51, %c128_i64 : i64
          llvm.store %52, %49 : i64, !llvm.ptr
          %53 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %54 = llvm.getelementptr %53[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %55 = llvm.load %54 : !llvm.ptr -> i64
          %56 = arith.muli %55, %c656_i64 : i64
          %57 = arith.index_cast %56 : i64 to index
          %58 = arith.divui %57, %c656 : index
          %alloc = memref.alloc(%58) : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
          %59 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %60 = llvm.getelementptr %59[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %61 = llvm.load %60 : !llvm.ptr -> i64
          %62 = arith.muli %61, %c32_i64 : i64
          %63 = arith.index_cast %62 : i64 to index
          %64 = arith.divui %63, %c32 : index
          %alloc_10 = memref.alloc(%64) : memref<?x4xf64>
          %65 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %66 = llvm.getelementptr %65[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %67 = llvm.load %66 : !llvm.ptr -> i64
          %68 = arith.muli %67, %c8_i64 : i64
          %69 = arith.index_cast %68 : i64 to index
          %70 = arith.divui %69, %c8 : index
          %alloc_11 = memref.alloc(%70) : memref<?xf64>
          %71 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %72 = llvm.getelementptr %71[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %73 = llvm.load %72 : !llvm.ptr -> i64
          %74 = arith.muli %73, %c32_i64 : i64
          %75 = arith.index_cast %74 : i64 to index
          %76 = arith.divui %75, %c32 : index
          %alloc_12 = memref.alloc(%76) : memref<?x4xf64>
          %77 = "polygeist.memref2pointer"(%alloc) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
          %78 = llvm.mlir.zero : !llvm.ptr
          %79 = llvm.icmp "eq" %77, %78 : !llvm.ptr
          %80 = scf.if %79 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_10) : (memref<?x4xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %81 = scf.if %80 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_11) : (memref<?xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %82 = scf.if %81 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_12) : (memref<?x4xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %83 = arith.xori %82, %true : i1
          %84 = arith.select %82, %c1_i32, %33 : i32
          scf.if %82 {
            %85 = llvm.mlir.addressof @str6 : !llvm.ptr
            %86 = llvm.getelementptr %85[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12 x i8>
            %87 = llvm.call @printf(%86) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
            memref.dealloc %alloc_10 : memref<?x4xf64>
            memref.dealloc %alloc_11 : memref<?xf64>
            memref.dealloc %alloc_12 : memref<?x4xf64>
            memref.dealloc %alloc : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
          }
          scf.yield %83, %84, %alloc_12, %alloc_11, %alloc_10, %alloc : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
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
      %17 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %18 = llvm.getelementptr %17[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %19 = "polygeist.memref2pointer"(%15#5) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
      %20:2 = scf.while (%arg2 = %c0_i32, %arg3 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %132 = llvm.load %18 : !llvm.ptr -> i32
        %133 = arith.cmpi slt, %arg2, %132 : i32
        scf.condition(%133) %arg2, %arg3 : i32, i32
      } do {
      ^bb0(%arg2: i32, %arg3: i32):
        %132:2 = scf.while (%arg4 = %c0_i32, %arg5 = %arg3) : (i32, i32) -> (i32, i32) {
          %134 = llvm.load %18 : !llvm.ptr -> i32
          %135 = arith.cmpi slt, %arg4, %134 : i32
          scf.condition(%135) %arg5, %arg4 : i32, i32
        } do {
        ^bb0(%arg4: i32, %arg5: i32):
          %134:2 = scf.while (%arg6 = %c0_i32, %arg7 = %arg4) : (i32, i32) -> (i32, i32) {
            %136 = llvm.load %18 : !llvm.ptr -> i32
            %137 = arith.cmpi slt, %arg6, %136 : i32
            scf.condition(%137) %arg7, %arg6 : i32, i32
          } do {
          ^bb0(%arg6: i32, %arg7: i32):
            %136 = arith.index_cast %arg6 : i32 to index
            %137 = arith.muli %136, %c656 : index
            %138 = arith.index_cast %137 : index to i64
            %139 = llvm.getelementptr %19[%138] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %140 = llvm.getelementptr %139[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            llvm.store %arg6, %140 : i32, !llvm.ptr
            %141 = llvm.getelementptr %139[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            %142 = arith.muli %arg6, %c128_i32 : i32
            %143 = arith.extsi %142 : i32 to i64
            llvm.store %143, %141 : i64, !llvm.ptr
            %144 = llvm.getelementptr %139[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            llvm.store %c0_i32, %144 : i32, !llvm.ptr
            scf.for %arg8 = %c-1 to %c2 step %c1 {
              %147 = arith.index_cast %arg8 : index to i32
              scf.for %arg9 = %c-1 to %c2 step %c1 {
                %148 = arith.index_cast %arg9 : index to i32
                scf.for %arg10 = %c-1 to %c2 step %c1 {
                  %149 = arith.index_cast %arg10 : index to i32
                  %150 = arith.cmpi eq, %147, %c0_i32 : i32
                  %151 = arith.cmpi eq, %148, %c0_i32 : i32
                  %152 = arith.andi %150, %151 : i1
                  %153 = scf.if %152 -> (i1) {
                    %160 = arith.cmpi eq, %149, %c0_i32 : i32
                    scf.yield %160 : i1
                  } else {
                    scf.yield %false : i1
                  }
                  %154 = arith.xori %153, %true : i1
                  %155 = arith.addi %arg2, %147 : i32
                  %156 = arith.addi %arg5, %148 : i32
                  %157 = arith.addi %arg7, %149 : i32
                  %158 = arith.cmpi sge, %155, %c0_i32 : i32
                  %159 = arith.andi %154, %158 : i1
                  scf.if %159 {
                    %160 = llvm.load %18 : !llvm.ptr -> i32
                    %161 = arith.cmpi slt, %155, %160 : i32
                    %162 = arith.cmpi sge, %156, %c0_i32 : i32
                    %163 = arith.andi %161, %162 : i1
                    scf.if %163 {
                      %164 = llvm.load %18 : !llvm.ptr -> i32
                      %165 = arith.cmpi slt, %156, %164 : i32
                      %166 = arith.cmpi sge, %157, %c0_i32 : i32
                      %167 = arith.andi %165, %166 : i1
                      scf.if %167 {
                        %168 = llvm.load %18 : !llvm.ptr -> i32
                        %169 = arith.cmpi slt, %157, %168 : i32
                        scf.if %169 {
                          %170 = llvm.load %144 : !llvm.ptr -> i32
                          %171 = arith.addi %170, %c1_i32 : i32
                          llvm.store %171, %144 : i32, !llvm.ptr
                          %172 = llvm.load %18 : !llvm.ptr -> i32
                          %173 = arith.muli %155, %172 : i32
                          %174 = arith.muli %173, %172 : i32
                          %175 = arith.muli %156, %172 : i32
                          %176 = arith.addi %174, %175 : i32
                          %177 = arith.addi %176, %157 : i32
                          %178 = llvm.getelementptr %139[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
                          %179 = arith.index_cast %170 : i32 to index
                          %180 = arith.muli %179, %c24 : index
                          %181 = arith.index_cast %180 : index to i64
                          %182 = llvm.getelementptr %178[%181] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                          %183 = llvm.getelementptr %182[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
                          llvm.store %177, %183 : i32, !llvm.ptr
                          %184 = llvm.getelementptr %182[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
                          %185 = arith.muli %177, %c128_i32 : i32
                          %186 = arith.extsi %185 : i32 to i64
                          llvm.store %186, %184 : i64, !llvm.ptr
                        }
                      }
                    }
                  }
                }
              }
            }
            %145 = arith.addi %arg6, %c1_i32 : i32
            %146 = arith.addi %arg7, %c1_i32 : i32
            scf.yield %146, %145 : i32, i32
          }
          %135 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %135, %134#0 : i32, i32
        }
        %133 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %133, %132#0 : i32, i32
      }
      func.call @srand(%c2_i32) : (i32) -> ()
      %21 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %22 = llvm.getelementptr %21[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %23 = scf.while (%arg2 = %c0_i64) : (i64) -> i64 {
        %132 = llvm.load %22 : !llvm.ptr -> i64
        %133 = arith.cmpi slt, %arg2, %132 : i64
        scf.condition(%133) %arg2 : i64
      } do {
      ^bb0(%arg2: i64):
        %132 = arith.index_cast %arg2 : i64 to index
        %133 = func.call @rand() : () -> i32
        %134 = arith.remsi %133, %c10_i32 : i32
        %135 = arith.addi %134, %c1_i32 : i32
        %136 = arith.sitofp %135 : i32 to f64
        %137 = arith.divf %136, %cst_5 : f64
        memref.store %137, %15#4[%132, %c0] : memref<?x4xf64>
        %138 = func.call @rand() : () -> i32
        %139 = arith.remsi %138, %c10_i32 : i32
        %140 = arith.addi %139, %c1_i32 : i32
        %141 = arith.sitofp %140 : i32 to f64
        %142 = arith.divf %141, %cst_5 : f64
        memref.store %142, %15#4[%132, %c1] : memref<?x4xf64>
        %143 = func.call @rand() : () -> i32
        %144 = arith.remsi %143, %c10_i32 : i32
        %145 = arith.addi %144, %c1_i32 : i32
        %146 = arith.sitofp %145 : i32 to f64
        %147 = arith.divf %146, %cst_5 : f64
        memref.store %147, %15#4[%132, %c2] : memref<?x4xf64>
        %148 = func.call @rand() : () -> i32
        %149 = arith.remsi %148, %c10_i32 : i32
        %150 = arith.addi %149, %c1_i32 : i32
        %151 = arith.sitofp %150 : i32 to f64
        %152 = arith.divf %151, %cst_5 : f64
        memref.store %152, %15#4[%132, %c3] : memref<?x4xf64>
        %153 = func.call @rand() : () -> i32
        %154 = arith.remsi %153, %c10_i32 : i32
        %155 = arith.addi %154, %c1_i32 : i32
        %156 = arith.sitofp %155 : i32 to f64
        %157 = arith.divf %156, %cst_5 : f64
        memref.store %157, %15#3[%132] : memref<?xf64>
        memref.store %cst_4, %15#2[%132, %c3] : memref<?x4xf64>
        %158 = memref.load %15#2[%132, %c3] : memref<?x4xf64>
        memref.store %158, %15#2[%132, %c2] : memref<?x4xf64>
        %159 = memref.load %15#2[%132, %c2] : memref<?x4xf64>
        memref.store %159, %15#2[%132, %c1] : memref<?x4xf64>
        %160 = memref.load %15#2[%132, %c1] : memref<?x4xf64>
        memref.store %160, %15#2[%132, %c0] : memref<?x4xf64>
        %161 = arith.addi %arg2, %c1_i64 : i64
        scf.yield %161 : i64
      }
      %cast = memref.cast %alloca_7 : memref<1x2xi64> to memref<?x2xi64>
      %24 = func.call @clock_gettime(%c1_i32, %cast) : (i32, memref<?x2xi64>) -> i32
      %25 = affine.load %alloca_9[] : memref<f64>
      %26 = arith.mulf %25, %cst_3 : f64
      %27 = arith.mulf %26, %25 : f64
      %28 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %29 = llvm.getelementptr %28[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %30 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %132 = arith.extsi %arg2 : i32 to i64
        %133 = llvm.load %29 : !llvm.ptr -> i64
        %134 = arith.cmpi slt, %132, %133 : i64
        scf.condition(%134) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        func.call @process_home_box(%arg2, %27, %15#5, %15#4, %15#3, %15#2) : (i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> ()
        %132 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %132 : i32
      }
      %cast_10 = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
      %31 = func.call @clock_gettime(%c1_i32, %cast_10) : (i32, memref<?x2xi64>) -> i32
      %32 = affine.load %alloca[0, 0] : memref<1x2xi64>
      %33 = affine.load %alloca_7[0, 0] : memref<1x2xi64>
      %34 = arith.subi %32, %33 : i64
      %35 = arith.sitofp %34 : i64 to f64
      %36 = arith.mulf %35, %cst_2 : f64
      %37 = affine.load %alloca[0, 1] : memref<1x2xi64>
      %38 = affine.load %alloca_7[0, 1] : memref<1x2xi64>
      %39 = arith.subi %37, %38 : i64
      %40 = arith.sitofp %39 : i64 to f64
      %41 = arith.divf %40, %cst_1 : f64
      %42 = arith.addf %36, %41 : f64
      %43 = llvm.mlir.addressof @str7 : !llvm.ptr
      %44 = llvm.getelementptr %43[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31 x i8>
      %45 = llvm.call @printf(%44, %42) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      %46 = llvm.mlir.addressof @str8 : !llvm.ptr
      %47 = llvm.getelementptr %46[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
      %48 = llvm.call @printf(%47) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %49 = llvm.mlir.addressof @str9 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
      %51 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %52 = llvm.getelementptr %51[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %53 = llvm.load %52 : !llvm.ptr -> i64
      %54 = llvm.call @printf(%50, %53) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      %55 = llvm.mlir.addressof @str10 : !llvm.ptr
      %56 = llvm.getelementptr %55[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
      %57 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %58 = llvm.getelementptr %57[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %59 = llvm.load %58 : !llvm.ptr -> i64
      %60 = llvm.call @printf(%56, %59) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      %61 = llvm.mlir.addressof @str11 : !llvm.ptr
      %62 = llvm.getelementptr %61[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
      %63 = llvm.call @printf(%62, %c128_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %64 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %65 = llvm.getelementptr %64[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %66 = llvm.load %65 : !llvm.ptr -> i64
      %67 = arith.index_cast %66 : i64 to index
      %68:12 = scf.for %arg2 = %c0 to %67 step %c1 iter_args(%arg3 = %cst, %arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_4, %arg12 = %cst_4, %arg13 = %cst_4, %arg14 = %cst_4) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        %132 = memref.load %15#2[%arg2, %c0] : memref<?x4xf64>
        %133 = arith.addf %arg14, %132 : f64
        %134 = memref.load %15#2[%arg2, %c1] : memref<?x4xf64>
        %135 = arith.addf %arg13, %134 : f64
        %136 = memref.load %15#2[%arg2, %c2] : memref<?x4xf64>
        %137 = arith.addf %arg12, %136 : f64
        %138 = memref.load %15#2[%arg2, %c3] : memref<?x4xf64>
        %139 = arith.addf %arg11, %138 : f64
        %140 = arith.cmpf ogt, %132, %arg10 : f64
        %141 = scf.if %140 -> (f64) {
          %156 = memref.load %15#2[%arg2, %c0] : memref<?x4xf64>
          scf.yield %156 : f64
        } else {
          scf.yield %arg10 : f64
        }
        %142 = arith.cmpf ogt, %134, %arg9 : f64
        %143 = scf.if %142 -> (f64) {
          %156 = memref.load %15#2[%arg2, %c1] : memref<?x4xf64>
          scf.yield %156 : f64
        } else {
          scf.yield %arg9 : f64
        }
        %144 = arith.cmpf ogt, %136, %arg8 : f64
        %145 = scf.if %144 -> (f64) {
          %156 = memref.load %15#2[%arg2, %c2] : memref<?x4xf64>
          scf.yield %156 : f64
        } else {
          scf.yield %arg8 : f64
        }
        %146 = arith.cmpf ogt, %138, %arg7 : f64
        %147 = scf.if %146 -> (f64) {
          %156 = memref.load %15#2[%arg2, %c3] : memref<?x4xf64>
          scf.yield %156 : f64
        } else {
          scf.yield %arg7 : f64
        }
        %148 = arith.cmpf olt, %132, %arg6 : f64
        %149 = scf.if %148 -> (f64) {
          %156 = memref.load %15#2[%arg2, %c0] : memref<?x4xf64>
          scf.yield %156 : f64
        } else {
          scf.yield %arg6 : f64
        }
        %150 = arith.cmpf olt, %134, %arg5 : f64
        %151 = scf.if %150 -> (f64) {
          %156 = memref.load %15#2[%arg2, %c1] : memref<?x4xf64>
          scf.yield %156 : f64
        } else {
          scf.yield %arg5 : f64
        }
        %152 = arith.cmpf olt, %136, %arg4 : f64
        %153 = scf.if %152 -> (f64) {
          %156 = memref.load %15#2[%arg2, %c2] : memref<?x4xf64>
          scf.yield %156 : f64
        } else {
          scf.yield %arg4 : f64
        }
        %154 = arith.cmpf olt, %138, %arg3 : f64
        %155 = scf.if %154 -> (f64) {
          %156 = memref.load %15#2[%arg2, %c3] : memref<?x4xf64>
          scf.yield %156 : f64
        } else {
          scf.yield %arg3 : f64
        }
        scf.yield %155, %153, %151, %149, %147, %145, %143, %141, %139, %137, %135, %133 : f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
      }
      %69 = llvm.mlir.addressof @str12 : !llvm.ptr
      %70 = llvm.getelementptr %69[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
      %71 = llvm.call @printf(%70) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %72 = llvm.mlir.addressof @str13 : !llvm.ptr
      %73 = llvm.getelementptr %72[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
      %74 = llvm.call @printf(%73) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %75 = llvm.mlir.addressof @str14 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<65 x i8>
      %77 = llvm.call @printf(%76) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %78 = llvm.mlir.addressof @str15 : !llvm.ptr
      %79 = llvm.getelementptr %78[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
      %80 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %81 = llvm.getelementptr %80[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %82 = llvm.load %81 : !llvm.ptr -> i64
      %83 = arith.sitofp %82 : i64 to f64
      %84 = arith.divf %68#11, %83 : f64
      %85 = llvm.call @printf(%79, %84, %68#3, %68#7) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %86 = llvm.mlir.addressof @str16 : !llvm.ptr
      %87 = llvm.getelementptr %86[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %88 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %89 = llvm.getelementptr %88[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %90 = llvm.load %89 : !llvm.ptr -> i64
      %91 = arith.sitofp %90 : i64 to f64
      %92 = arith.divf %68#10, %91 : f64
      %93 = llvm.call @printf(%87, %92, %68#2, %68#6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %94 = llvm.mlir.addressof @str17 : !llvm.ptr
      %95 = llvm.getelementptr %94[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %96 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %97 = llvm.getelementptr %96[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %98 = llvm.load %97 : !llvm.ptr -> i64
      %99 = arith.sitofp %98 : i64 to f64
      %100 = arith.divf %68#9, %99 : f64
      %101 = llvm.call @printf(%95, %100, %68#1, %68#5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %102 = llvm.mlir.addressof @str18 : !llvm.ptr
      %103 = llvm.getelementptr %102[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %104 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %105 = llvm.getelementptr %104[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %106 = llvm.load %105 : !llvm.ptr -> i64
      %107 = arith.sitofp %106 : i64 to f64
      %108 = arith.divf %68#8, %107 : f64
      %109 = llvm.call @printf(%103, %108, %68#0, %68#4) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %110 = llvm.mlir.addressof @str19 : !llvm.ptr
      %111 = llvm.getelementptr %110[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
      %112 = llvm.call @printf(%111) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %113 = llvm.mlir.addressof @str20 : !llvm.ptr
      %114 = llvm.getelementptr %113[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<74 x i8>
      %115 = llvm.call @printf(%114) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %116 = llvm.mlir.addressof @str21 : !llvm.ptr
      %117 = llvm.getelementptr %116[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<75 x i8>
      %118 = llvm.call @printf(%117) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %119 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %120 = llvm.getelementptr %119[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %121 = llvm.mlir.addressof @str22 : !llvm.ptr
      %122 = llvm.getelementptr %121[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
      %123 = scf.while (%arg2 = %c0_i64) : (i64) -> i64 {
        %132 = llvm.load %120 : !llvm.ptr -> i64
        %133 = arith.cmpi slt, %arg2, %132 : i64
        scf.condition(%133) %arg2 : i64
      } do {
      ^bb0(%arg2: i64):
        %132 = arith.index_cast %arg2 : i64 to index
        %133 = memref.load %15#2[%132, %c0] : memref<?x4xf64>
        %134 = memref.load %15#2[%132, %c1] : memref<?x4xf64>
        %135 = memref.load %15#2[%132, %c2] : memref<?x4xf64>
        %136 = memref.load %15#2[%132, %c3] : memref<?x4xf64>
        %137 = llvm.call @printf(%122, %arg2, %133, %134, %135, %136) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64, f64, f64, f64) -> i32
        %138 = arith.addi %arg2, %c1_i64 : i64
        scf.yield %138 : i64
      }
      %124 = llvm.mlir.addressof @str23 : !llvm.ptr
      %125 = llvm.mlir.addressof @str24 : !llvm.ptr
      %126 = "polygeist.pointer2memref"(%124) : (!llvm.ptr) -> memref<?xi8>
      %127 = "polygeist.pointer2memref"(%125) : (!llvm.ptr) -> memref<?xi8>
      %128 = func.call @fopen(%126, %127) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %129 = "polygeist.memref2pointer"(%128) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %130 = llvm.mlir.zero : !llvm.ptr
      %131 = llvm.icmp "ne" %129, %130 : !llvm.ptr
      scf.if %131 {
        %132 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %133 = llvm.getelementptr %132[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %134 = "polygeist.pointer2memref"(%133) : (!llvm.ptr) -> memref<?xi8>
        %135 = func.call @fwrite(%134, %c8_i64, %c1_i64, %128) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %136 = arith.cmpi ne, %135, %c1_i64 : i64
        scf.if %136 {
          %151 = llvm.mlir.addressof @str25 : !llvm.ptr
          %152 = llvm.getelementptr %151[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
          %153 = llvm.call @printf(%152) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %137 = "polygeist.memref2pointer"(%15#2) : (memref<?x4xf64>) -> !llvm.ptr
        %138 = "polygeist.pointer2memref"(%137) : (!llvm.ptr) -> memref<?xi8>
        %139 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %140 = llvm.getelementptr %139[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %141 = llvm.load %140 : !llvm.ptr -> i64
        %142 = func.call @fwrite(%138, %c32_i64, %141, %128) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %143 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %144 = llvm.getelementptr %143[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %145 = llvm.load %144 : !llvm.ptr -> i64
        %146 = arith.cmpi ne, %142, %145 : i64
        scf.if %146 {
          %151 = llvm.mlir.addressof @str26 : !llvm.ptr
          %152 = llvm.getelementptr %151[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
          %153 = llvm.call @printf(%152) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %147 = func.call @fclose(%128) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
        %148 = llvm.mlir.addressof @str27 : !llvm.ptr
        %149 = llvm.getelementptr %148[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31 x i8>
        %150 = llvm.call @printf(%149) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      } else {
        %132 = llvm.mlir.addressof @str28 : !llvm.ptr
        %133 = llvm.getelementptr %132[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
        %134 = llvm.call @printf(%133) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
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
  func.func private @clock_gettime(i32, memref<?x2xi64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @process_home_box(%arg0: i32, %arg1: f64, %arg2: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c656 = arith.constant 656 : index
    %cst = arith.constant 1.000000e+02 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
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
      affine.store %cst_0, %alloca[0, 0] : memref<1x4xf64>
      affine.store %cst_0, %alloca[0, 1] : memref<1x4xf64>
      affine.store %cst_0, %alloca[0, 2] : memref<1x4xf64>
      affine.store %cst_0, %alloca[0, 3] : memref<1x4xf64>
      %cast = memref.cast %alloca : memref<1x4xf64> to memref<?x4xf64>
      %10 = arith.index_cast %9 : i32 to index
      %11 = memref.load %arg4[%10] : memref<?xf64>
      %12 = arith.mulf %11, %cst : f64
      %13 = arith.fptosi %12 : f64 to i32
      func.call @self_box_accumulate(%9, %7, %arg1, %arg3, %arg4, %cast, %13) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> ()
      %14 = llvm.getelementptr %4[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %15 = llvm.load %14 : !llvm.ptr -> i32
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
  func.func private @self_box_accumulate(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg7 = %c0 to %c128 step %c1 {
      %0 = arith.index_cast %arg7 : index to i32
      %1 = arith.addi %arg1, %0 : i32
      %2 = arith.cmpi ne, %arg0, %1 : i32
      scf.if %2 {
        %3 = func.call @pair_interaction(%arg0, %1, %arg2, %arg3, %arg4, %arg5) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
      }
    }
    return
  }
  func.func private @neighbor_box_accumulate(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg4: memref<?x4xf64>, %arg5: memref<?xf64>, %arg6: memref<?x4xf64>, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c656 = arith.constant 656 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.muli %0, %c656 : index
    %2 = arith.index_cast %1 : index to i64
    %3 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
    %4 = llvm.getelementptr %3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %5 = llvm.getelementptr %4[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %6 = llvm.getelementptr %4[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %7 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
      %8 = llvm.load %5 : !llvm.ptr -> i32
      %9 = arith.cmpi slt, %arg8, %8 : i32
      scf.condition(%9) %arg8 : i32
    } do {
    ^bb0(%arg8: i32):
      %8 = arith.index_cast %arg8 : i32 to index
      %9 = arith.muli %8, %c24 : index
      %10 = arith.index_cast %9 : index to i64
      %11 = llvm.getelementptr %6[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %12 = llvm.getelementptr %11[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
      %13 = llvm.load %12 : !llvm.ptr -> i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = arith.muli %14, %c656 : index
      %16 = arith.index_cast %15 : index to i64
      %17 = llvm.getelementptr %3[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %18 = llvm.getelementptr %17[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %19 = llvm.load %18 : !llvm.ptr -> i64
      %20 = arith.trunci %19 : i64 to i32
      scf.for %arg9 = %c0 to %c128 step %c1 {
        %22 = arith.index_cast %arg9 : index to i32
        %23 = arith.addi %20, %22 : i32
        %24 = func.call @pair_interaction(%arg0, %23, %arg2, %arg4, %arg5, %arg6) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
      }
      %21 = arith.addi %arg8, %c1_i32 : i32
      scf.yield %21 : i32
    }
    return
  }
}
