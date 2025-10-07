module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  // A) pairwise interaction substitution (func_substitute)
  // decision: 0 = exact, 1 = approximate
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "pair_interaction",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 200>,   // state in [0, 200+] (u² * 100)
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 129>,           // u² < 0.70 => exact; else substitute
    decisions = array<i32: 0, 1>
  }> : () -> ()

  // Required for func_substitute
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "pair_interaction"}> : () -> ()

  // B) self-box inner loop perforation (loop_perforate)
  // decisions: 1 = step 1 (exact), 2 = step 2 (perforated)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "self_box_accumulate",
    transform_type = "loop_perforate",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 100>,   // state in [0,100] (= |q_i| * 100)
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 35>,           // |q_i| >= 0.60 => step=1 else step=2
    decisions = array<i32: 0, 0>
  }> : () -> ()

  // C) neighbor-box perforation (loop_perforate over k/j)
  // decisions: 1 = step 1 (exact), 2 = step 2 (perforated)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "neighbor_box_accumulate",
    transform_type = "loop_perforate",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 26>,    // state = nn in [0,26]
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 5>,           // nn >= 13 => step=1 else step=2
    decisions = array<i32: 1, 0>
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
  llvm.mlir.global internal constant @str7("Total execution time: %f seconds\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: OOM\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("Configuration: boxes1d = %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("Usage: %s -boxes1d <number>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("ERROR: Usage: %s -boxes1d <number>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("ERROR: -boxes1d > 0\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("-boxes1d\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("WG size of kernel = %d\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @pair_interaction(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>, %arg6: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %cst = arith.constant 0x42008E8D71C00000 : f64
    %cst_0 = arith.constant 0xC2008E8D71C00000 : f64
    %cst_1 = arith.constant 1.000000e+06 : f64
    %cst_2 = arith.constant 2.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %cst_3 = arith.constant 0.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+01 : f64
    %c10_i32 = arith.constant 10 : i32
    %c0_i64 = arith.constant 0 : i64
    %c2_i32 = arith.constant 2 : i32
    %c128_i64 = arith.constant 128 : i64
    %cst_5 = arith.constant 5.000000e-01 : f64
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
    %alloca_6 = memref.alloca() : memref<f64>
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
          affine.store %cst_5, %alloca_6[] : memref<f64>
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
          %alloc_7 = memref.alloc(%64) : memref<?x4xf64>
          %65 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %66 = llvm.getelementptr %65[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %67 = llvm.load %66 : !llvm.ptr -> i64
          %68 = arith.muli %67, %c8_i64 : i64
          %69 = arith.index_cast %68 : i64 to index
          %70 = arith.divui %69, %c8 : index
          %alloc_8 = memref.alloc(%70) : memref<?xf64>
          %71 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %72 = llvm.getelementptr %71[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %73 = llvm.load %72 : !llvm.ptr -> i64
          %74 = arith.muli %73, %c32_i64 : i64
          %75 = arith.index_cast %74 : i64 to index
          %76 = arith.divui %75, %c32 : index
          %alloc_9 = memref.alloc(%76) : memref<?x4xf64>
          %77 = "polygeist.memref2pointer"(%alloc) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
          %78 = llvm.mlir.zero : !llvm.ptr
          %79 = llvm.icmp "eq" %77, %78 : !llvm.ptr
          %80 = scf.if %79 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_7) : (memref<?x4xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %81 = scf.if %80 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_8) : (memref<?xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %82 = scf.if %81 -> (i1) {
            scf.yield %true : i1
          } else {
            %85 = "polygeist.memref2pointer"(%alloc_9) : (memref<?x4xf64>) -> !llvm.ptr
            %86 = llvm.icmp "eq" %85, %78 : !llvm.ptr
            scf.yield %86 : i1
          }
          %83 = arith.xori %82, %true : i1
          %84 = arith.select %82, %c1_i32, %33 : i32
          scf.if %82 {
            %85 = llvm.mlir.addressof @str6 : !llvm.ptr
            %86 = llvm.getelementptr %85[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12 x i8>
            %87 = llvm.call @printf(%86) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
            memref.dealloc %alloc_7 : memref<?x4xf64>
            memref.dealloc %alloc_8 : memref<?xf64>
            memref.dealloc %alloc_9 : memref<?x4xf64>
            memref.dealloc %alloc : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
          }
          scf.yield %83, %84, %alloc_9, %alloc_8, %alloc_7, %alloc : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
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
        %124 = llvm.load %18 : !llvm.ptr -> i32
        %125 = arith.cmpi slt, %arg2, %124 : i32
        scf.condition(%125) %arg2, %arg3 : i32, i32
      } do {
      ^bb0(%arg2: i32, %arg3: i32):
        %124:2 = scf.while (%arg4 = %c0_i32, %arg5 = %arg3) : (i32, i32) -> (i32, i32) {
          %126 = llvm.load %18 : !llvm.ptr -> i32
          %127 = arith.cmpi slt, %arg4, %126 : i32
          scf.condition(%127) %arg5, %arg4 : i32, i32
        } do {
        ^bb0(%arg4: i32, %arg5: i32):
          %126:2 = scf.while (%arg6 = %c0_i32, %arg7 = %arg4) : (i32, i32) -> (i32, i32) {
            %128 = llvm.load %18 : !llvm.ptr -> i32
            %129 = arith.cmpi slt, %arg6, %128 : i32
            scf.condition(%129) %arg7, %arg6 : i32, i32
          } do {
          ^bb0(%arg6: i32, %arg7: i32):
            %128 = arith.index_cast %arg6 : i32 to index
            %129 = arith.muli %128, %c656 : index
            %130 = arith.index_cast %129 : index to i64
            %131 = llvm.getelementptr %19[%130] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %132 = llvm.getelementptr %131[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            llvm.store %arg6, %132 : i32, !llvm.ptr
            %133 = llvm.getelementptr %131[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            %134 = arith.muli %arg6, %c128_i32 : i32
            %135 = arith.extsi %134 : i32 to i64
            llvm.store %135, %133 : i64, !llvm.ptr
            %136 = llvm.getelementptr %131[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            llvm.store %c0_i32, %136 : i32, !llvm.ptr
            scf.for %arg8 = %c-1 to %c2 step %c1 {
              %139 = arith.index_cast %arg8 : index to i32
              scf.for %arg9 = %c-1 to %c2 step %c1 {
                %140 = arith.index_cast %arg9 : index to i32
                scf.for %arg10 = %c-1 to %c2 step %c1 {
                  %141 = arith.index_cast %arg10 : index to i32
                  %142 = arith.cmpi eq, %139, %c0_i32 : i32
                  %143 = arith.cmpi eq, %140, %c0_i32 : i32
                  %144 = arith.andi %142, %143 : i1
                  %145 = scf.if %144 -> (i1) {
                    %152 = arith.cmpi eq, %141, %c0_i32 : i32
                    scf.yield %152 : i1
                  } else {
                    scf.yield %false : i1
                  }
                  %146 = arith.xori %145, %true : i1
                  %147 = arith.addi %arg2, %139 : i32
                  %148 = arith.addi %arg5, %140 : i32
                  %149 = arith.addi %arg7, %141 : i32
                  %150 = arith.cmpi sge, %147, %c0_i32 : i32
                  %151 = arith.andi %146, %150 : i1
                  scf.if %151 {
                    %152 = llvm.load %18 : !llvm.ptr -> i32
                    %153 = arith.cmpi slt, %147, %152 : i32
                    %154 = arith.cmpi sge, %148, %c0_i32 : i32
                    %155 = arith.andi %153, %154 : i1
                    scf.if %155 {
                      %156 = llvm.load %18 : !llvm.ptr -> i32
                      %157 = arith.cmpi slt, %148, %156 : i32
                      %158 = arith.cmpi sge, %149, %c0_i32 : i32
                      %159 = arith.andi %157, %158 : i1
                      scf.if %159 {
                        %160 = llvm.load %18 : !llvm.ptr -> i32
                        %161 = arith.cmpi slt, %149, %160 : i32
                        scf.if %161 {
                          %162 = llvm.load %136 : !llvm.ptr -> i32
                          %163 = arith.addi %162, %c1_i32 : i32
                          llvm.store %163, %136 : i32, !llvm.ptr
                          %164 = llvm.load %18 : !llvm.ptr -> i32
                          %165 = arith.muli %147, %164 : i32
                          %166 = arith.muli %165, %164 : i32
                          %167 = arith.muli %148, %164 : i32
                          %168 = arith.addi %166, %167 : i32
                          %169 = arith.addi %168, %149 : i32
                          %170 = llvm.getelementptr %131[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
                          %171 = arith.index_cast %162 : i32 to index
                          %172 = arith.muli %171, %c24 : index
                          %173 = arith.index_cast %172 : index to i64
                          %174 = llvm.getelementptr %170[%173] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                          %175 = llvm.getelementptr %174[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
                          llvm.store %169, %175 : i32, !llvm.ptr
                          %176 = llvm.getelementptr %174[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
                          %177 = arith.muli %169, %c128_i32 : i32
                          %178 = arith.extsi %177 : i32 to i64
                          llvm.store %178, %176 : i64, !llvm.ptr
                        }
                      }
                    }
                  }
                }
              }
            }
            %137 = arith.addi %arg6, %c1_i32 : i32
            %138 = arith.addi %arg7, %c1_i32 : i32
            scf.yield %138, %137 : i32, i32
          }
          %127 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %127, %126#0 : i32, i32
        }
        %125 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %125, %124#0 : i32, i32
      }
      func.call @srand(%c2_i32) : (i32) -> ()
      %21 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %22 = llvm.getelementptr %21[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %23 = scf.while (%arg2 = %c0_i64) : (i64) -> i64 {
        %124 = llvm.load %22 : !llvm.ptr -> i64
        %125 = arith.cmpi slt, %arg2, %124 : i64
        scf.condition(%125) %arg2 : i64
      } do {
      ^bb0(%arg2: i64):
        %124 = arith.index_cast %arg2 : i64 to index
        %125 = func.call @rand() : () -> i32
        %126 = arith.remsi %125, %c10_i32 : i32
        %127 = arith.addi %126, %c1_i32 : i32
        %128 = arith.sitofp %127 : i32 to f64
        %129 = arith.divf %128, %cst_4 : f64
        memref.store %129, %15#4[%124, %c0] : memref<?x4xf64>
        %130 = func.call @rand() : () -> i32
        %131 = arith.remsi %130, %c10_i32 : i32
        %132 = arith.addi %131, %c1_i32 : i32
        %133 = arith.sitofp %132 : i32 to f64
        %134 = arith.divf %133, %cst_4 : f64
        memref.store %134, %15#4[%124, %c1] : memref<?x4xf64>
        %135 = func.call @rand() : () -> i32
        %136 = arith.remsi %135, %c10_i32 : i32
        %137 = arith.addi %136, %c1_i32 : i32
        %138 = arith.sitofp %137 : i32 to f64
        %139 = arith.divf %138, %cst_4 : f64
        memref.store %139, %15#4[%124, %c2] : memref<?x4xf64>
        %140 = func.call @rand() : () -> i32
        %141 = arith.remsi %140, %c10_i32 : i32
        %142 = arith.addi %141, %c1_i32 : i32
        %143 = arith.sitofp %142 : i32 to f64
        %144 = arith.divf %143, %cst_4 : f64
        memref.store %144, %15#4[%124, %c3] : memref<?x4xf64>
        %145 = func.call @rand() : () -> i32
        %146 = arith.remsi %145, %c10_i32 : i32
        %147 = arith.addi %146, %c1_i32 : i32
        %148 = arith.sitofp %147 : i32 to f64
        %149 = arith.divf %148, %cst_4 : f64
        memref.store %149, %15#3[%124] : memref<?xf64>
        memref.store %cst_3, %15#2[%124, %c3] : memref<?x4xf64>
        %150 = memref.load %15#2[%124, %c3] : memref<?x4xf64>
        memref.store %150, %15#2[%124, %c2] : memref<?x4xf64>
        %151 = memref.load %15#2[%124, %c2] : memref<?x4xf64>
        memref.store %151, %15#2[%124, %c1] : memref<?x4xf64>
        %152 = memref.load %15#2[%124, %c1] : memref<?x4xf64>
        memref.store %152, %15#2[%124, %c0] : memref<?x4xf64>
        %153 = arith.addi %arg2, %c1_i64 : i64
        scf.yield %153 : i64
      }
      %24 = func.call @get_time() : () -> i64
      %25 = affine.load %alloca_6[] : memref<f64>
      %26 = arith.mulf %25, %cst_2 : f64
      %27 = arith.mulf %26, %25 : f64
      %28 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %29 = llvm.getelementptr %28[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %30 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %124 = arith.extsi %arg2 : i32 to i64
        %125 = llvm.load %29 : !llvm.ptr -> i64
        %126 = arith.cmpi slt, %124, %125 : i64
        scf.condition(%126) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        func.call @process_home_box(%arg2, %27, %15#5, %15#4, %15#3, %15#2) : (i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> ()
        %124 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %124 : i32
      }
      %31 = func.call @get_time() : () -> i64
      %32 = llvm.mlir.addressof @str7 : !llvm.ptr
      %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
      %34 = arith.subi %31, %24 : i64
      %35 = arith.sitofp %34 : i64 to f64
      %36 = arith.divf %35, %cst_1 : f64
      %37 = llvm.call @printf(%33, %36) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      %38 = llvm.mlir.addressof @str8 : !llvm.ptr
      %39 = llvm.getelementptr %38[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
      %40 = llvm.call @printf(%39) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %41 = llvm.mlir.addressof @str9 : !llvm.ptr
      %42 = llvm.getelementptr %41[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
      %43 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %44 = llvm.getelementptr %43[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %45 = llvm.load %44 : !llvm.ptr -> i64
      %46 = llvm.call @printf(%42, %45) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      %47 = llvm.mlir.addressof @str10 : !llvm.ptr
      %48 = llvm.getelementptr %47[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
      %49 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %50 = llvm.getelementptr %49[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %51 = llvm.load %50 : !llvm.ptr -> i64
      %52 = llvm.call @printf(%48, %51) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      %53 = llvm.mlir.addressof @str11 : !llvm.ptr
      %54 = llvm.getelementptr %53[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
      %55 = llvm.call @printf(%54, %c128_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %56 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %57 = llvm.getelementptr %56[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %58 = llvm.load %57 : !llvm.ptr -> i64
      %59 = arith.index_cast %58 : i64 to index
      %60:12 = scf.for %arg2 = %c0 to %59 step %c1 iter_args(%arg3 = %cst, %arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_3, %arg12 = %cst_3, %arg13 = %cst_3, %arg14 = %cst_3) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        %124 = memref.load %15#2[%arg2, %c0] : memref<?x4xf64>
        %125 = arith.addf %arg14, %124 : f64
        %126 = memref.load %15#2[%arg2, %c1] : memref<?x4xf64>
        %127 = arith.addf %arg13, %126 : f64
        %128 = memref.load %15#2[%arg2, %c2] : memref<?x4xf64>
        %129 = arith.addf %arg12, %128 : f64
        %130 = memref.load %15#2[%arg2, %c3] : memref<?x4xf64>
        %131 = arith.addf %arg11, %130 : f64
        %132 = arith.cmpf ogt, %124, %arg10 : f64
        %133 = scf.if %132 -> (f64) {
          %148 = memref.load %15#2[%arg2, %c0] : memref<?x4xf64>
          scf.yield %148 : f64
        } else {
          scf.yield %arg10 : f64
        }
        %134 = arith.cmpf ogt, %126, %arg9 : f64
        %135 = scf.if %134 -> (f64) {
          %148 = memref.load %15#2[%arg2, %c1] : memref<?x4xf64>
          scf.yield %148 : f64
        } else {
          scf.yield %arg9 : f64
        }
        %136 = arith.cmpf ogt, %128, %arg8 : f64
        %137 = scf.if %136 -> (f64) {
          %148 = memref.load %15#2[%arg2, %c2] : memref<?x4xf64>
          scf.yield %148 : f64
        } else {
          scf.yield %arg8 : f64
        }
        %138 = arith.cmpf ogt, %130, %arg7 : f64
        %139 = scf.if %138 -> (f64) {
          %148 = memref.load %15#2[%arg2, %c3] : memref<?x4xf64>
          scf.yield %148 : f64
        } else {
          scf.yield %arg7 : f64
        }
        %140 = arith.cmpf olt, %124, %arg6 : f64
        %141 = scf.if %140 -> (f64) {
          %148 = memref.load %15#2[%arg2, %c0] : memref<?x4xf64>
          scf.yield %148 : f64
        } else {
          scf.yield %arg6 : f64
        }
        %142 = arith.cmpf olt, %126, %arg5 : f64
        %143 = scf.if %142 -> (f64) {
          %148 = memref.load %15#2[%arg2, %c1] : memref<?x4xf64>
          scf.yield %148 : f64
        } else {
          scf.yield %arg5 : f64
        }
        %144 = arith.cmpf olt, %128, %arg4 : f64
        %145 = scf.if %144 -> (f64) {
          %148 = memref.load %15#2[%arg2, %c2] : memref<?x4xf64>
          scf.yield %148 : f64
        } else {
          scf.yield %arg4 : f64
        }
        %146 = arith.cmpf olt, %130, %arg3 : f64
        %147 = scf.if %146 -> (f64) {
          %148 = memref.load %15#2[%arg2, %c3] : memref<?x4xf64>
          scf.yield %148 : f64
        } else {
          scf.yield %arg3 : f64
        }
        scf.yield %147, %145, %143, %141, %139, %137, %135, %133, %131, %129, %127, %125 : f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
      }
      %61 = llvm.mlir.addressof @str12 : !llvm.ptr
      %62 = llvm.getelementptr %61[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
      %63 = llvm.call @printf(%62) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %64 = llvm.mlir.addressof @str13 : !llvm.ptr
      %65 = llvm.getelementptr %64[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
      %66 = llvm.call @printf(%65) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %67 = llvm.mlir.addressof @str14 : !llvm.ptr
      %68 = llvm.getelementptr %67[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<65 x i8>
      %69 = llvm.call @printf(%68) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %70 = llvm.mlir.addressof @str15 : !llvm.ptr
      %71 = llvm.getelementptr %70[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
      %72 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %73 = llvm.getelementptr %72[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %74 = llvm.load %73 : !llvm.ptr -> i64
      %75 = arith.sitofp %74 : i64 to f64
      %76 = arith.divf %60#11, %75 : f64
      %77 = llvm.call @printf(%71, %76, %60#3, %60#7) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %78 = llvm.mlir.addressof @str16 : !llvm.ptr
      %79 = llvm.getelementptr %78[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %80 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %81 = llvm.getelementptr %80[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %82 = llvm.load %81 : !llvm.ptr -> i64
      %83 = arith.sitofp %82 : i64 to f64
      %84 = arith.divf %60#10, %83 : f64
      %85 = llvm.call @printf(%79, %84, %60#2, %60#6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %86 = llvm.mlir.addressof @str17 : !llvm.ptr
      %87 = llvm.getelementptr %86[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %88 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %89 = llvm.getelementptr %88[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %90 = llvm.load %89 : !llvm.ptr -> i64
      %91 = arith.sitofp %90 : i64 to f64
      %92 = arith.divf %60#9, %91 : f64
      %93 = llvm.call @printf(%87, %92, %60#1, %60#5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %94 = llvm.mlir.addressof @str18 : !llvm.ptr
      %95 = llvm.getelementptr %94[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %96 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %97 = llvm.getelementptr %96[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %98 = llvm.load %97 : !llvm.ptr -> i64
      %99 = arith.sitofp %98 : i64 to f64
      %100 = arith.divf %60#8, %99 : f64
      %101 = llvm.call @printf(%95, %100, %60#0, %60#4) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %102 = llvm.mlir.addressof @str19 : !llvm.ptr
      %103 = llvm.getelementptr %102[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
      %104 = llvm.call @printf(%103) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %105 = llvm.mlir.addressof @str20 : !llvm.ptr
      %106 = llvm.getelementptr %105[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<74 x i8>
      %107 = llvm.call @printf(%106) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %108 = llvm.mlir.addressof @str21 : !llvm.ptr
      %109 = llvm.getelementptr %108[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<75 x i8>
      %110 = llvm.call @printf(%109) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %111 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %112 = llvm.getelementptr %111[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %113 = llvm.mlir.addressof @str22 : !llvm.ptr
      %114 = llvm.getelementptr %113[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
      %115 = scf.while (%arg2 = %c0_i64) : (i64) -> i64 {
        %124 = llvm.load %112 : !llvm.ptr -> i64
        %125 = arith.cmpi slt, %arg2, %124 : i64
        scf.condition(%125) %arg2 : i64
      } do {
      ^bb0(%arg2: i64):
        %124 = arith.index_cast %arg2 : i64 to index
        %125 = memref.load %15#2[%124, %c0] : memref<?x4xf64>
        %126 = memref.load %15#2[%124, %c1] : memref<?x4xf64>
        %127 = memref.load %15#2[%124, %c2] : memref<?x4xf64>
        %128 = memref.load %15#2[%124, %c3] : memref<?x4xf64>
        %129 = llvm.call @printf(%114, %arg2, %125, %126, %127, %128) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64, f64, f64, f64) -> i32
        %130 = arith.addi %arg2, %c1_i64 : i64
        scf.yield %130 : i64
      }
      %116 = llvm.mlir.addressof @str23 : !llvm.ptr
      %117 = llvm.mlir.addressof @str24 : !llvm.ptr
      %118 = "polygeist.pointer2memref"(%116) : (!llvm.ptr) -> memref<?xi8>
      %119 = "polygeist.pointer2memref"(%117) : (!llvm.ptr) -> memref<?xi8>
      %120 = func.call @fopen(%118, %119) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %121 = "polygeist.memref2pointer"(%120) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %122 = llvm.mlir.zero : !llvm.ptr
      %123 = llvm.icmp "ne" %121, %122 : !llvm.ptr
      scf.if %123 {
        %124 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %125 = llvm.getelementptr %124[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %126 = "polygeist.pointer2memref"(%125) : (!llvm.ptr) -> memref<?xi8>
        %127 = func.call @fwrite(%126, %c8_i64, %c1_i64, %120) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %128 = arith.cmpi ne, %127, %c1_i64 : i64
        scf.if %128 {
          %143 = llvm.mlir.addressof @str25 : !llvm.ptr
          %144 = llvm.getelementptr %143[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
          %145 = llvm.call @printf(%144) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %129 = "polygeist.memref2pointer"(%15#2) : (memref<?x4xf64>) -> !llvm.ptr
        %130 = "polygeist.pointer2memref"(%129) : (!llvm.ptr) -> memref<?xi8>
        %131 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %132 = llvm.getelementptr %131[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %133 = llvm.load %132 : !llvm.ptr -> i64
        %134 = func.call @fwrite(%130, %c32_i64, %133, %120) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %135 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %136 = llvm.getelementptr %135[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %137 = llvm.load %136 : !llvm.ptr -> i64
        %138 = arith.cmpi ne, %134, %137 : i64
        scf.if %138 {
          %143 = llvm.mlir.addressof @str26 : !llvm.ptr
          %144 = llvm.getelementptr %143[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
          %145 = llvm.call @printf(%144) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %139 = func.call @fclose(%120) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
        %140 = llvm.mlir.addressof @str27 : !llvm.ptr
        %141 = llvm.getelementptr %140[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31 x i8>
        %142 = llvm.call @printf(%141) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      } else {
        %124 = llvm.mlir.addressof @str28 : !llvm.ptr
        %125 = llvm.getelementptr %124[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
        %126 = llvm.call @printf(%125) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
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
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    scf.for %arg7 = %c0 to %c128 step %c1 {
      %1 = arith.index_cast %arg7 : index to i32
      %2 = arith.addi %arg1, %1 : i32
      %3 = arith.cmpi ne, %arg0, %2 : i32
      scf.if %3 {
        %4 = affine.load %arg3[symbol(%0), 0] : memref<?x4xf64>
        %5 = arith.index_cast %2 : i32 to index
        %6 = memref.load %arg3[%5, %c0] : memref<?x4xf64>
        %7 = arith.addf %4, %6 : f64
        %8 = affine.load %arg3[symbol(%0), 1] : memref<?x4xf64>
        %9 = memref.load %arg3[%5, %c1] : memref<?x4xf64>
        %10 = arith.mulf %8, %9 : f64
        %11 = affine.load %arg3[symbol(%0), 2] : memref<?x4xf64>
        %12 = memref.load %arg3[%5, %c2] : memref<?x4xf64>
        %13 = arith.mulf %11, %12 : f64
        %14 = arith.addf %10, %13 : f64
        %15 = affine.load %arg3[symbol(%0), 3] : memref<?x4xf64>
        %16 = memref.load %arg3[%5, %c3] : memref<?x4xf64>
        %17 = arith.mulf %15, %16 : f64
        %18 = arith.addf %14, %17 : f64
        %19 = arith.subf %7, %18 : f64
        %20 = arith.cmpf olt, %19, %cst : f64
        %21 = arith.select %20, %cst, %19 : f64
        %22 = arith.mulf %arg2, %21 : f64
        %23 = func.call @state_pair_from_u2(%22) : (f64) -> i32
        %24 = func.call @pair_interaction(%arg0, %2, %arg2, %arg3, %arg4, %arg5, %23) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
      }
    }
    return
  }
  func.func private @neighbor_box_accumulate(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg4: memref<?x4xf64>, %arg5: memref<?xf64>, %arg6: memref<?x4xf64>, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c656 = arith.constant 656 : index
    %c24 = arith.constant 24 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.muli %1, %c656 : index
    %3 = arith.index_cast %2 : index to i64
    %4 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
    %5 = llvm.getelementptr %4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %6 = llvm.getelementptr %5[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %7 = llvm.getelementptr %5[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %8 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
      %9 = llvm.load %6 : !llvm.ptr -> i32
      %10 = arith.cmpi slt, %arg8, %9 : i32
      scf.condition(%10) %arg8 : i32
    } do {
    ^bb0(%arg8: i32):
      %9 = arith.index_cast %arg8 : i32 to index
      %10 = arith.muli %9, %c24 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = llvm.getelementptr %7[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %13 = llvm.getelementptr %12[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
      %14 = llvm.load %13 : !llvm.ptr -> i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %15, %c656 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = llvm.getelementptr %4[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %19 = llvm.getelementptr %18[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %20 = llvm.load %19 : !llvm.ptr -> i64
      %21 = arith.trunci %20 : i64 to i32
      scf.for %arg9 = %c0 to %c128 step %c1 {
        %23 = arith.index_cast %arg9 : index to i32
        %24 = arith.addi %21, %23 : i32
        %25 = affine.load %arg4[symbol(%0), 0] : memref<?x4xf64>
        %26 = arith.index_cast %24 : i32 to index
        %27 = memref.load %arg4[%26, %c0] : memref<?x4xf64>
        %28 = arith.addf %25, %27 : f64
        %29 = affine.load %arg4[symbol(%0), 1] : memref<?x4xf64>
        %30 = memref.load %arg4[%26, %c1] : memref<?x4xf64>
        %31 = arith.mulf %29, %30 : f64
        %32 = affine.load %arg4[symbol(%0), 2] : memref<?x4xf64>
        %33 = memref.load %arg4[%26, %c2] : memref<?x4xf64>
        %34 = arith.mulf %32, %33 : f64
        %35 = arith.addf %31, %34 : f64
        %36 = affine.load %arg4[symbol(%0), 3] : memref<?x4xf64>
        %37 = memref.load %arg4[%26, %c3] : memref<?x4xf64>
        %38 = arith.mulf %36, %37 : f64
        %39 = arith.addf %35, %38 : f64
        %40 = arith.subf %28, %39 : f64
        %41 = arith.cmpf olt, %40, %cst : f64
        %42 = arith.select %41, %cst, %40 : f64
        %43 = arith.mulf %arg2, %42 : f64
        %44 = func.call @state_pair_from_u2(%43) : (f64) -> i32
        %45 = func.call @pair_interaction(%arg0, %24, %arg2, %arg4, %arg5, %arg6, %44) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> i32
      }
      %22 = arith.addi %arg8, %c1_i32 : i32
      scf.yield %22 : i32
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
