module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  // Knob — CNDF substitution (proper func_substitute)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "compute_cndf",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 2>,
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 0>,
    decisions = array<i32: 1, 1>
  }> : () -> ()


  // Required for func_substitute
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "compute_cndf"}> : () -> ()

  // Knob — Black-Scholes approximation (func_substitute)
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "BlkSchlsEqEuroNoDiv",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 2>,
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 0>,
    decisions = array<i32: 0, 0>
  }> : () -> ()

  // Required for func_substitute
  "approxMLIR.util.annotation.convert_to_call"() <{func_name = "BlkSchlsEqEuroNoDiv"}> : () -> ()
  llvm.mlir.global internal constant @str33("%zu %d %.17g %.17g %.17g %.17g %.17g %.17g\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str32("# idx otype sptprice strike rate volatility otime price\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str31("fopen text dump failed\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str30("w\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str29("%s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str28(".txt\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str27("alloc txtPath failed\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str26("------------------------------------------------------------------------------------------------\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str25("%3zu | %13.6f | %13.6f | %13.6f | %13.6f | %13.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str24("----|---------------|---------------|---------------|---------------|---------------|--------------\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str23("  # |       S       |       K       |     Rate      |      Vol      |     Time      |    Price\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str22("\0A--- First 20 Option Prices ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str21("      Max Price: %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str20("      Min Price: %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str19("  Average Price: %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str18("\0A--- Price Statistics ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("Number of options: %zu\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("\0A--- Black-Scholes Results ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("alloc prices failed\00") {addr_space = 0 : i32}
  memref.global @prices : memref<1xmemref<?xf64>> = uninitialized
  memref.global @otime : memref<1xmemref<?xf64>> = uninitialized
  memref.global @volatility : memref<1xmemref<?xf64>> = uninitialized
  memref.global @rate : memref<1xmemref<?xf64>> = uninitialized
  memref.global @strike : memref<1xmemref<?xf64>> = uninitialized
  memref.global @sptprice : memref<1xmemref<?xf64>> = uninitialized
  memref.global @otype : memref<1xmemref<?xi32>> = uninitialized
  llvm.mlir.global internal constant @str14("ERROR: Unable to open file `%s`.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("rb\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("Error: <nthreads> must be 1 (serial version)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Usage:\0A\09%s <nthreads> <inputFile> <outputFile>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("PARSEC Benchmark Suite\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Elapsed: %f\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  memref.global @numOptions : memref<1xi64> = dense<0>
  memref.global "private" @const6 : memref<1xf64> = dense<1.3302744289999999>
  memref.global "private" @const5 : memref<1xf64> = dense<1.8212559779999999>
  memref.global "private" @const4 : memref<1xf64> = dense<1.781477937>
  memref.global "private" @const3 : memref<1xf64> = dense<0.356563782>
  memref.global "private" @const2 : memref<1xf64> = dense<0.31938153000000002>
  memref.global "private" @const1 : memref<1xf64> = dense<0.23164190000000001>
  memref.global "private" @one : memref<1xf64> = dense<1.000000e+00>
  memref.global "private" @inv_sqrt_2xPI : memref<1xf64> = dense<0.3989422804014327>
  memref.global "private" @half : memref<1xf64> = dense<5.000000e-01>
  memref.global "private" @zero : memref<1xf64> = dense<0.000000e+00>
  llvm.mlir.global internal constant @str8("Unsupported data type in file\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("alloc fail\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("Could not allocate\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("read error\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("File pointer is not valid\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Unsupported data type to write\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("write error\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("Could Not Open File for writing\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("wb\00") {addr_space = 0 : i32}
  func.func @writeQualityFile(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: i32, %arg3: i64) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8_i64 = arith.constant 8 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %alloca = memref.alloca() : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi32>
    affine.store %arg2, %alloca_0[0] : memref<1xi32>
    affine.store %arg3, %alloca[0] : memref<1xi64>
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi8>
    %2 = call @fopen(%arg0, %1) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %3 = "polygeist.memref2pointer"(%2) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.icmp "eq" %3, %4 : !llvm.ptr
    scf.if %5 {
      %17 = llvm.mlir.addressof @str1 : !llvm.ptr
      %18 = "polygeist.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%18) : (memref<?xi8>) -> ()
    }
    %6 = "polygeist.memref2pointer"(%alloca) : (memref<1xi64>) -> !llvm.ptr
    %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr) -> memref<?xi8>
    %8 = call @fwrite(%7, %c8_i64, %c1_i64, %2) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
    %9 = arith.cmpi ne, %8, %c1_i64 : i64
    scf.if %9 {
      %17 = llvm.mlir.addressof @str2 : !llvm.ptr
      %18 = "polygeist.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%18) : (memref<?xi8>) -> ()
    }
    %10 = "polygeist.memref2pointer"(%alloca_0) : (memref<1xi32>) -> !llvm.ptr
    %11 = "polygeist.pointer2memref"(%10) : (!llvm.ptr) -> memref<?xi8>
    %12 = call @fwrite(%11, %c4_i64, %c1_i64, %2) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
    %13 = arith.cmpi ne, %12, %c1_i64 : i64
    scf.if %13 {
      %17 = llvm.mlir.addressof @str2 : !llvm.ptr
      %18 = "polygeist.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%18) : (memref<?xi8>) -> ()
    }
    %14 = affine.load %alloca_0[0] : memref<1xi32>
    %15 = arith.cmpi eq, %14, %c0_i32 : i32
    scf.if %15 {
      %17 = affine.load %alloca[0] : memref<1xi64>
      %18 = func.call @fwrite(%arg1, %c8_i64, %17, %2) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
      %19 = arith.cmpi ne, %18, %17 : i64
      scf.if %19 {
        %20 = llvm.mlir.addressof @str2 : !llvm.ptr
        %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
        func.call @die(%21) : (memref<?xi8>) -> ()
      }
    } else {
      %17 = arith.cmpi eq, %14, %c1_i32 : i32
      scf.if %17 {
        %18 = affine.load %alloca[0] : memref<1xi64>
        %19 = func.call @fwrite(%arg1, %c4_i64, %18, %2) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %20 = arith.cmpi ne, %19, %18 : i64
        scf.if %20 {
          %21 = llvm.mlir.addressof @str2 : !llvm.ptr
          %22 = "polygeist.pointer2memref"(%21) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%22) : (memref<?xi8>) -> ()
        }
      } else {
        %18 = arith.cmpi eq, %14, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloca[0] : memref<1xi64>
          %20 = func.call @fwrite(%arg1, %c4_i64, %19, %2) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
          %21 = arith.cmpi ne, %20, %19 : i64
          scf.if %21 {
            %22 = llvm.mlir.addressof @str2 : !llvm.ptr
            %23 = "polygeist.pointer2memref"(%22) : (!llvm.ptr) -> memref<?xi8>
            func.call @die(%23) : (memref<?xi8>) -> ()
          }
        } else {
          %19 = llvm.mlir.addressof @str3 : !llvm.ptr
          %20 = "polygeist.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%20) : (memref<?xi8>) -> ()
        }
      }
    }
    %16 = call @fclose(%2) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
    return
  }
  func.func private @fopen(memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @die(%arg0: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %2 = "polygeist.memref2pointer"(%1) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %3 = llvm.mlir.addressof @str29 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %5 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %6 = llvm.call @fprintf(%2, %4, %5) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    call @exit(%c1_i32) : (i32) -> ()
    return
  }
  func.func private @fwrite(memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fclose(memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @readDataAsDouble(%arg0: memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, %arg1: memref<?xmemref<?xf64>>, %arg2: memref<?xi64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %alloca = memref.alloca() : memref<1xi32>
    %0 = llvm.mlir.undef : i32
    affine.store %0, %alloca[0] : memref<1xi32>
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    scf.if %3 {
      %20 = llvm.mlir.addressof @str4 : !llvm.ptr
      %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%21) : (memref<?xi8>) -> ()
    }
    %4 = "polygeist.memref2pointer"(%arg2) : (memref<?xi64>) -> !llvm.ptr
    %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr) -> memref<?xi8>
    %6 = call @fread(%5, %c8_i64, %c1_i64, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
    %7 = arith.cmpi ne, %6, %c1_i64 : i64
    scf.if %7 {
      %20 = llvm.mlir.addressof @str5 : !llvm.ptr
      %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%21) : (memref<?xi8>) -> ()
    }
    %8 = affine.load %arg2[0] : memref<?xi64>
    %9 = arith.muli %8, %c8_i64 : i64
    %10 = arith.index_cast %9 : i64 to index
    %11 = arith.divui %10, %c8 : index
    %alloc = memref.alloc(%11) : memref<?xf64>
    %12 = "polygeist.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
    %13 = llvm.icmp "eq" %12, %2 : !llvm.ptr
    scf.if %13 {
      %20 = llvm.mlir.addressof @str6 : !llvm.ptr
      %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%21) : (memref<?xi8>) -> ()
    }
    affine.store %alloc, %arg1[0] : memref<?xmemref<?xf64>>
    %14 = "polygeist.memref2pointer"(%alloca) : (memref<1xi32>) -> !llvm.ptr
    %15 = "polygeist.pointer2memref"(%14) : (!llvm.ptr) -> memref<?xi8>
    %16 = call @fread(%15, %c4_i64, %c1_i64, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
    %17 = arith.cmpi ne, %16, %c1_i64 : i64
    scf.if %17 {
      %20 = llvm.mlir.addressof @str5 : !llvm.ptr
      %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%21) : (memref<?xi8>) -> ()
    }
    %18 = affine.load %alloca[0] : memref<1xi32>
    %19 = arith.cmpi eq, %18, %c0_i32 : i32
    scf.if %19 {
      %20 = "polygeist.pointer2memref"(%12) : (!llvm.ptr) -> memref<?xi8>
      %21 = func.call @fread(%20, %c8_i64, %8, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
      %22 = arith.cmpi ne, %21, %8 : i64
      scf.if %22 {
        %23 = llvm.mlir.addressof @str5 : !llvm.ptr
        %24 = "polygeist.pointer2memref"(%23) : (!llvm.ptr) -> memref<?xi8>
        func.call @die(%24) : (memref<?xi8>) -> ()
      }
    } else {
      %20 = arith.cmpi eq, %18, %c1_i32 : i32
      scf.if %20 {
        %21 = arith.muli %8, %c4_i64 : i64
        %22 = arith.index_cast %21 : i64 to index
        %23 = arith.divui %22, %c4 : index
        %alloc_0 = memref.alloc(%23) : memref<?xf32>
        %24 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xf32>) -> !llvm.ptr
        %25 = llvm.icmp "eq" %24, %2 : !llvm.ptr
        scf.if %25 {
          %31 = llvm.mlir.addressof @str7 : !llvm.ptr
          %32 = "polygeist.pointer2memref"(%31) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%32) : (memref<?xi8>) -> ()
        }
        %26 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xf32>) -> !llvm.ptr
        %27 = "polygeist.pointer2memref"(%26) : (!llvm.ptr) -> memref<?xi8>
        %28 = func.call @fread(%27, %c4_i64, %8, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %29 = arith.cmpi ne, %28, %8 : i64
        scf.if %29 {
          %31 = llvm.mlir.addressof @str5 : !llvm.ptr
          %32 = "polygeist.pointer2memref"(%31) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%32) : (memref<?xi8>) -> ()
        }
        %30 = arith.index_cast %8 : i64 to index
        scf.for %arg3 = %c0 to %30 step %c1 {
          %31 = memref.load %alloc_0[%arg3] : memref<?xf32>
          %32 = arith.extf %31 : f32 to f64
          memref.store %32, %alloc[%arg3] : memref<?xf64>
        }
        memref.dealloc %alloc_0 : memref<?xf32>
      } else {
        %21 = arith.cmpi eq, %18, %c2_i32 : i32
        scf.if %21 {
          %22 = arith.muli %8, %c4_i64 : i64
          %23 = arith.index_cast %22 : i64 to index
          %24 = arith.divui %23, %c4 : index
          %alloc_0 = memref.alloc(%24) : memref<?xi32>
          %25 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xi32>) -> !llvm.ptr
          %26 = llvm.icmp "eq" %25, %2 : !llvm.ptr
          scf.if %26 {
            %32 = llvm.mlir.addressof @str7 : !llvm.ptr
            %33 = "polygeist.pointer2memref"(%32) : (!llvm.ptr) -> memref<?xi8>
            func.call @die(%33) : (memref<?xi8>) -> ()
          }
          %27 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xi32>) -> !llvm.ptr
          %28 = "polygeist.pointer2memref"(%27) : (!llvm.ptr) -> memref<?xi8>
          %29 = func.call @fread(%28, %c4_i64, %8, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
          %30 = arith.cmpi ne, %29, %8 : i64
          scf.if %30 {
            %32 = llvm.mlir.addressof @str5 : !llvm.ptr
            %33 = "polygeist.pointer2memref"(%32) : (!llvm.ptr) -> memref<?xi8>
            func.call @die(%33) : (memref<?xi8>) -> ()
          }
          %31 = arith.index_cast %8 : i64 to index
          scf.for %arg3 = %c0 to %31 step %c1 {
            %32 = memref.load %alloc_0[%arg3] : memref<?xi32>
            %33 = arith.sitofp %32 : i32 to f64
            memref.store %33, %alloc[%arg3] : memref<?xf64>
          }
          memref.dealloc %alloc_0 : memref<?xi32>
        } else {
          %22 = llvm.mlir.addressof @str8 : !llvm.ptr
          %23 = "polygeist.pointer2memref"(%22) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%23) : (memref<?xi8>) -> ()
        }
      }
    }
    return
  }
  func.func private @fread(memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @readDataAsInt(%arg0: memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, %arg1: memref<?xmemref<?xi32>>, %arg2: memref<?xi64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %alloca = memref.alloca() : memref<1xi32>
    %0 = llvm.mlir.undef : i32
    affine.store %0, %alloca[0] : memref<1xi32>
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    scf.if %3 {
      %20 = llvm.mlir.addressof @str4 : !llvm.ptr
      %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%21) : (memref<?xi8>) -> ()
    }
    %4 = "polygeist.memref2pointer"(%arg2) : (memref<?xi64>) -> !llvm.ptr
    %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr) -> memref<?xi8>
    %6 = call @fread(%5, %c8_i64, %c1_i64, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
    %7 = arith.cmpi ne, %6, %c1_i64 : i64
    scf.if %7 {
      %20 = llvm.mlir.addressof @str5 : !llvm.ptr
      %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%21) : (memref<?xi8>) -> ()
    }
    %8 = affine.load %arg2[0] : memref<?xi64>
    %9 = arith.muli %8, %c4_i64 : i64
    %10 = arith.index_cast %9 : i64 to index
    %11 = arith.divui %10, %c4 : index
    %alloc = memref.alloc(%11) : memref<?xi32>
    %12 = "polygeist.memref2pointer"(%alloc) : (memref<?xi32>) -> !llvm.ptr
    %13 = llvm.icmp "eq" %12, %2 : !llvm.ptr
    scf.if %13 {
      %20 = llvm.mlir.addressof @str6 : !llvm.ptr
      %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%21) : (memref<?xi8>) -> ()
    }
    affine.store %alloc, %arg1[0] : memref<?xmemref<?xi32>>
    %14 = "polygeist.memref2pointer"(%alloca) : (memref<1xi32>) -> !llvm.ptr
    %15 = "polygeist.pointer2memref"(%14) : (!llvm.ptr) -> memref<?xi8>
    %16 = call @fread(%15, %c4_i64, %c1_i64, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
    %17 = arith.cmpi ne, %16, %c1_i64 : i64
    scf.if %17 {
      %20 = llvm.mlir.addressof @str5 : !llvm.ptr
      %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%21) : (memref<?xi8>) -> ()
    }
    %18 = affine.load %alloca[0] : memref<1xi32>
    %19 = arith.cmpi eq, %18, %c2_i32 : i32
    scf.if %19 {
      %20 = "polygeist.pointer2memref"(%12) : (!llvm.ptr) -> memref<?xi8>
      %21 = func.call @fread(%20, %c4_i64, %8, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
      %22 = arith.cmpi ne, %21, %8 : i64
      scf.if %22 {
        %23 = llvm.mlir.addressof @str5 : !llvm.ptr
        %24 = "polygeist.pointer2memref"(%23) : (!llvm.ptr) -> memref<?xi8>
        func.call @die(%24) : (memref<?xi8>) -> ()
      }
    } else {
      %20 = arith.cmpi eq, %18, %c0_i32 : i32
      scf.if %20 {
        %21 = arith.muli %8, %c8_i64 : i64
        %22 = arith.index_cast %21 : i64 to index
        %23 = arith.divui %22, %c8 : index
        %alloc_0 = memref.alloc(%23) : memref<?xf64>
        %24 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xf64>) -> !llvm.ptr
        %25 = llvm.icmp "eq" %24, %2 : !llvm.ptr
        scf.if %25 {
          %31 = llvm.mlir.addressof @str7 : !llvm.ptr
          %32 = "polygeist.pointer2memref"(%31) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%32) : (memref<?xi8>) -> ()
        }
        %26 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xf64>) -> !llvm.ptr
        %27 = "polygeist.pointer2memref"(%26) : (!llvm.ptr) -> memref<?xi8>
        %28 = func.call @fread(%27, %c8_i64, %8, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %29 = arith.cmpi ne, %28, %8 : i64
        scf.if %29 {
          %31 = llvm.mlir.addressof @str5 : !llvm.ptr
          %32 = "polygeist.pointer2memref"(%31) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%32) : (memref<?xi8>) -> ()
        }
        %30 = arith.index_cast %8 : i64 to index
        scf.for %arg3 = %c0 to %30 step %c1 {
          %31 = memref.load %alloc_0[%arg3] : memref<?xf64>
          %32 = arith.fptosi %31 : f64 to i32
          memref.store %32, %alloc[%arg3] : memref<?xi32>
        }
        memref.dealloc %alloc_0 : memref<?xf64>
      } else {
        %21 = arith.cmpi eq, %18, %c1_i32 : i32
        scf.if %21 {
          %alloc_0 = memref.alloc(%11) : memref<?xf32>
          %22 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xf32>) -> !llvm.ptr
          %23 = llvm.icmp "eq" %22, %2 : !llvm.ptr
          scf.if %23 {
            %29 = llvm.mlir.addressof @str7 : !llvm.ptr
            %30 = "polygeist.pointer2memref"(%29) : (!llvm.ptr) -> memref<?xi8>
            func.call @die(%30) : (memref<?xi8>) -> ()
          }
          %24 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xf32>) -> !llvm.ptr
          %25 = "polygeist.pointer2memref"(%24) : (!llvm.ptr) -> memref<?xi8>
          %26 = func.call @fread(%25, %c4_i64, %8, %arg0) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
          %27 = arith.cmpi ne, %26, %8 : i64
          scf.if %27 {
            %29 = llvm.mlir.addressof @str5 : !llvm.ptr
            %30 = "polygeist.pointer2memref"(%29) : (!llvm.ptr) -> memref<?xi8>
            func.call @die(%30) : (memref<?xi8>) -> ()
          }
          %28 = arith.index_cast %8 : i64 to index
          scf.for %arg3 = %c0 to %28 step %c1 {
            %29 = memref.load %alloc_0[%arg3] : memref<?xf32>
            %30 = arith.fptosi %29 : f32 to i32
            memref.store %30, %alloc[%arg3] : memref<?xi32>
          }
          memref.dealloc %alloc_0 : memref<?xf32>
        } else {
          %22 = llvm.mlir.addressof @str8 : !llvm.ptr
          %23 = "polygeist.pointer2memref"(%22) : (!llvm.ptr) -> memref<?xi8>
          func.call @die(%23) : (memref<?xi8>) -> ()
        }
      }
    }
    return
  }
  func.func @compute_cndf(%arg0: f64, %arg1: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = memref.get_global @zero : memref<1xf64>
    %1 = affine.load %0[0] : memref<1xf64>
    %2 = arith.cmpf olt, %arg0, %1 : f64
    %3 = scf.if %2 -> (f64) {
      %50 = arith.negf %arg0 : f64
      scf.yield %50 : f64
    } else {
      scf.yield %arg0 : f64
    }
    %4 = memref.get_global @half : memref<1xf64>
    %5 = affine.load %4[0] : memref<1xf64>
    %6 = arith.negf %5 : f64
    %7 = arith.mulf %6, %3 : f64
    %8 = arith.mulf %7, %3 : f64
    %9 = math.exp %8 : f64
    %10 = memref.get_global @inv_sqrt_2xPI : memref<1xf64>
    %11 = affine.load %10[0] : memref<1xf64>
    %12 = arith.mulf %9, %11 : f64
    %13 = memref.get_global @one : memref<1xf64>
    %14 = affine.load %13[0] : memref<1xf64>
    %15 = memref.get_global @const1 : memref<1xf64>
    %16 = affine.load %15[0] : memref<1xf64>
    %17 = arith.mulf %16, %3 : f64
    %18 = arith.addf %14, %17 : f64
    %19 = arith.divf %14, %18 : f64
    %20 = arith.mulf %19, %19 : f64
    %21 = arith.mulf %20, %19 : f64
    %22 = arith.mulf %21, %19 : f64
    %23 = arith.mulf %22, %19 : f64
    %24 = memref.get_global @const2 : memref<1xf64>
    %25 = affine.load %24[0] : memref<1xf64>
    %26 = arith.mulf %19, %25 : f64
    %27 = memref.get_global @const3 : memref<1xf64>
    %28 = affine.load %27[0] : memref<1xf64>
    %29 = arith.negf %28 : f64
    %30 = arith.mulf %20, %29 : f64
    %31 = arith.addf %26, %30 : f64
    %32 = memref.get_global @const4 : memref<1xf64>
    %33 = affine.load %32[0] : memref<1xf64>
    %34 = arith.mulf %21, %33 : f64
    %35 = arith.addf %31, %34 : f64
    %36 = memref.get_global @const5 : memref<1xf64>
    %37 = affine.load %36[0] : memref<1xf64>
    %38 = arith.negf %37 : f64
    %39 = arith.mulf %22, %38 : f64
    %40 = arith.addf %35, %39 : f64
    %41 = memref.get_global @const6 : memref<1xf64>
    %42 = affine.load %41[0] : memref<1xf64>
    %43 = arith.mulf %23, %42 : f64
    %44 = arith.addf %40, %43 : f64
    %45 = memref.get_global @one : memref<1xf64>
    %46 = affine.load %45[0] : memref<1xf64>
    %47 = arith.mulf %44, %12 : f64
    %48 = arith.subf %46, %47 : f64
    %49 = scf.if %2 -> (f64) {
      %50 = memref.get_global @one : memref<1xf64>
      %51 = affine.load %50[0] : memref<1xf64>
      %52 = arith.subf %51, %48 : f64
      scf.yield %52 : f64
    } else {
      scf.yield %48 : f64
    }
    return %49 : f64
  }
  func.func @approx_compute_cndf(%arg0: f64, %arg1: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = memref.get_global @zero : memref<1xf64>
    %1 = affine.load %0[0] : memref<1xf64>
    %2 = arith.cmpf olt, %arg0, %1 : f64
    %3 = scf.if %2 -> (f64) {
      %39 = arith.negf %arg0 : f64
      scf.yield %39 : f64
    } else {
      scf.yield %arg0 : f64
    }
    %4 = memref.get_global @half : memref<1xf64>
    %5 = affine.load %4[0] : memref<1xf64>
    %6 = arith.negf %5 : f64
    %7 = arith.mulf %6, %3 : f64
    %8 = arith.mulf %7, %3 : f64
    %9 = math.exp %8 : f64
    %10 = memref.get_global @inv_sqrt_2xPI : memref<1xf64>
    %11 = affine.load %10[0] : memref<1xf64>
    %12 = arith.mulf %9, %11 : f64
    %13 = memref.get_global @one : memref<1xf64>
    %14 = affine.load %13[0] : memref<1xf64>
    %15 = memref.get_global @const1 : memref<1xf64>
    %16 = affine.load %15[0] : memref<1xf64>
    %17 = arith.mulf %16, %3 : f64
    %18 = arith.addf %14, %17 : f64
    %19 = arith.divf %14, %18 : f64
    %20 = arith.mulf %19, %19 : f64
    %21 = arith.mulf %20, %19 : f64
    %22 = memref.get_global @const2 : memref<1xf64>
    %23 = affine.load %22[0] : memref<1xf64>
    %24 = arith.mulf %19, %23 : f64
    %25 = memref.get_global @const3 : memref<1xf64>
    %26 = affine.load %25[0] : memref<1xf64>
    %27 = arith.negf %26 : f64
    %28 = arith.mulf %20, %27 : f64
    %29 = arith.addf %24, %28 : f64
    %30 = memref.get_global @const4 : memref<1xf64>
    %31 = affine.load %30[0] : memref<1xf64>
    %32 = arith.mulf %21, %31 : f64
    %33 = arith.addf %29, %32 : f64
    %34 = memref.get_global @one : memref<1xf64>
    %35 = affine.load %34[0] : memref<1xf64>
    %36 = arith.mulf %33, %12 : f64
    %37 = arith.subf %35, %36 : f64
    %38 = scf.if %2 -> (f64) {
      %39 = memref.get_global @one : memref<1xf64>
      %40 = affine.load %39[0] : memref<1xf64>
      %41 = arith.subf %40, %37 : f64
      scf.yield %41 : f64
    } else {
      scf.yield %37 : f64
    }
    return %38 : f64
  }
  func.func @approx_BlkSchlsEqEuroNoDiv(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64, %arg5: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e-01 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %cst_1 = arith.constant 3.000000e-01 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %0 = math.sqrt %arg4 : f64
    %1 = arith.divf %arg0, %arg1 : f64
    %2 = arith.subf %1, %cst_2 : f64
    %3 = math.absf %2 : f64
    %4 = arith.cmpf olt, %3, %cst_1 : f64
    %5 = scf.if %4 -> (f64) {
      %29 = arith.mulf %2, %2 : f64
      %30 = arith.mulf %29, %cst_0 : f64
      %31 = arith.subf %2, %30 : f64
      scf.yield %31 : f64
    } else {
      %29 = math.log %1 : f64
      scf.yield %29 : f64
    }
    %6 = arith.mulf %arg3, %arg3 : f64
    %7 = memref.get_global @half : memref<1xf64>
    %8 = affine.load %7[0] : memref<1xf64>
    %9 = arith.mulf %6, %8 : f64
    %10 = arith.addf %arg2, %9 : f64
    %11 = arith.mulf %10, %arg4 : f64
    %12 = arith.addf %11, %5 : f64
    %13 = arith.mulf %arg3, %0 : f64
    %14 = arith.divf %12, %13 : f64
    %15 = arith.mulf %arg3, %0 : f64
    %16 = arith.subf %14, %15 : f64
    %17 = arith.mulf %arg2, %arg4 : f64
    %18 = arith.cmpf olt, %17, %cst : f64
    %19 = scf.if %18 -> (f64) {
      %29 = arith.subf %cst_2, %17 : f64
      %30 = arith.mulf %29, %arg1 : f64
      scf.yield %30 : f64
    } else {
      %29 = arith.negf %arg2 : f64
      %30 = arith.mulf %29, %arg4 : f64
      %31 = math.exp %30 : f64
      %32 = arith.mulf %31, %arg1 : f64
      scf.yield %32 : f64
    }
    %20 = call @decide_cndf_state(%14) : (f64) -> i32
    %21 = call @decide_cndf_state(%16) : (f64) -> i32
    %22 = call @compute_cndf(%14, %20) : (f64, i32) -> f64
    %23 = call @compute_cndf(%16, %20) : (f64, i32) -> f64
    %24 = arith.subf %cst_2, %23 : f64
    %25 = arith.mulf %19, %24 : f64
    %26 = arith.subf %cst_2, %22 : f64
    %27 = arith.mulf %arg0, %26 : f64
    %28 = arith.subf %25, %27 : f64
    return %28 : f64
  }
  func.func private @decide_cndf_state(%arg0: f64) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 1.000000e+00 : f64
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant 2.000000e+00 : f64
    %0 = math.absf %arg0 : f64
    %1 = arith.cmpf ogt, %0, %cst_0 : f64
    %2 = scf.if %1 -> (i32) {
      scf.yield %c2_i32 : i32
    } else {
      %3 = arith.cmpf ogt, %0, %cst : f64
      %4 = arith.extui %3 : i1 to i32
      scf.yield %4 : i32
    }
    return %2 : i32
  }
  func.func @bs_thread() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+06 : f64
    %c0_i64 = arith.constant 0 : i64
    %0 = call @clock() : () -> i64
    %1 = memref.get_global @numOptions : memref<1xi64>
    %2 = affine.load %1[0] : memref<1xi64>
    call @price_range(%c0_i64, %2) : (i64, i64) -> ()
    %3 = call @clock() : () -> i64
    %4 = arith.subi %3, %0 : i64
    %5 = arith.sitofp %4 : i64 to f64
    %6 = arith.divf %5, %cst : f64
    %7 = llvm.mlir.addressof @str9 : !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<13 x i8>
    %9 = llvm.call @printf(%8, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    return %c0_i32 : i32
  }
  func.func private @clock() -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @price_range(%arg0: i64, %arg1: i64) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg1 : i64 to index
    %1 = arith.index_cast %arg0 : i64 to index
    scf.for %arg2 = %1 to %0 step %c1 {
      %2 = arith.subi %arg2, %1 : index
      %3 = arith.index_cast %arg0 : i64 to index
      %4 = arith.addi %3, %2 : index
      %5 = memref.get_global @sptprice : memref<1xmemref<?xf64>>
      %6 = affine.load %5[0] : memref<1xmemref<?xf64>>
      %7 = memref.load %6[%4] : memref<?xf64>
      %8 = memref.get_global @strike : memref<1xmemref<?xf64>>
      %9 = affine.load %8[0] : memref<1xmemref<?xf64>>
      %10 = memref.load %9[%4] : memref<?xf64>
      %11 = memref.get_global @rate : memref<1xmemref<?xf64>>
      %12 = affine.load %11[0] : memref<1xmemref<?xf64>>
      %13 = memref.load %12[%4] : memref<?xf64>
      %14 = memref.get_global @otime : memref<1xmemref<?xf64>>
      %15 = affine.load %14[0] : memref<1xmemref<?xf64>>
      %16 = memref.load %15[%4] : memref<?xf64>
      %17 = func.call @decide_bs_state(%7, %10, %13, %16) : (f64, f64, f64, f64) -> i32
      %18 = memref.get_global @prices : memref<1xmemref<?xf64>>
      %19 = affine.load %18[0] : memref<1xmemref<?xf64>>
      %20 = affine.load %5[0] : memref<1xmemref<?xf64>>
      %21 = memref.load %20[%4] : memref<?xf64>
      %22 = affine.load %8[0] : memref<1xmemref<?xf64>>
      %23 = memref.load %22[%4] : memref<?xf64>
      %24 = affine.load %11[0] : memref<1xmemref<?xf64>>
      %25 = memref.load %24[%4] : memref<?xf64>
      %26 = memref.get_global @volatility : memref<1xmemref<?xf64>>
      %27 = affine.load %26[0] : memref<1xmemref<?xf64>>
      %28 = memref.load %27[%4] : memref<?xf64>
      %29 = affine.load %14[0] : memref<1xmemref<?xf64>>
      %30 = memref.load %29[%4] : memref<?xf64>
      %31 = func.call @BlkSchlsEqEuroNoDiv(%21, %23, %25, %28, %30, %17) : (f64, f64, f64, f64, f64, i32) -> f64
      memref.store %31, %19[%4] : memref<?xf64>
    }
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c5 = arith.constant 5 : index
    %c5_i64 = arith.constant 5 : i64
    %c20_i64 = arith.constant 20 : i64
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>>
    %alloca_0 = memref.alloca() : memref<memref<?xi8>>
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.addressof @str10 : !llvm.ptr
    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
    %3 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %4 = arith.cmpi ne, %arg0, %c4_i32 : i32
    %5 = arith.cmpi eq, %arg0, %c4_i32 : i32
    %6 = arith.select %4, %c1_i32, %0 : i32
    scf.if %4 {
      %9 = llvm.mlir.addressof @str11 : !llvm.ptr
      %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<48 x i8>
      %11 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %12 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
      %13 = llvm.call @printf(%10, %12) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    %7:2 = scf.if %5 -> (i1, i32) {
      %9 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %10 = func.call @atoi(%9) : (memref<?xi8>) -> i32
      %11 = arith.cmpi eq, %10, %c1_i32 : i32
      %12 = arith.cmpi ne, %10, %c1_i32 : i32
      %13 = arith.select %12, %c1_i32, %6 : i32
      scf.if %12 {
        %15 = llvm.mlir.addressof @str12 : !llvm.ptr
        %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<46 x i8>
        %17 = llvm.call @printf(%16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      }
      %14:2 = scf.if %11 -> (i1, i32) {
        %15 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %16 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
        affine.store %16, %alloca_0[] : memref<memref<?xi8>>
        %17 = llvm.mlir.addressof @str13 : !llvm.ptr
        %18 = "polygeist.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xi8>
        %19 = func.call @fopen(%15, %18) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
        affine.store %19, %alloca[] : memref<memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>>
        %20 = "polygeist.memref2pointer"(%19) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
        %21 = llvm.mlir.zero : !llvm.ptr
        %22 = llvm.icmp "eq" %20, %21 : !llvm.ptr
        %23 = arith.xori %22, %true : i1
        %24 = arith.select %22, %c1_i32, %13 : i32
        scf.if %22 {
          %25 = llvm.mlir.addressof @str14 : !llvm.ptr
          %26 = llvm.getelementptr %25[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
          %27 = "polygeist.memref2pointer"(%15) : (memref<?xi8>) -> !llvm.ptr
          %28 = llvm.call @printf(%26, %27) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        }
        scf.yield %23, %24 : i1, i32
      } else {
        scf.yield %false, %6 : i1, i32
      }
      scf.yield %14#0, %14#1 : i1, i32
    } else {
      scf.yield %false, %6 : i1, i32
    }
    %8 = arith.select %7#0, %c0_i32, %7#1 : i32
    scf.if %7#0 {
      %9 = affine.load %alloca[] : memref<memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>>
      %10 = memref.get_global @otype : memref<1xmemref<?xi32>>
      %cast = memref.cast %10 : memref<1xmemref<?xi32>> to memref<?xmemref<?xi32>>
      %11 = memref.get_global @numOptions : memref<1xi64>
      %cast_1 = memref.cast %11 : memref<1xi64> to memref<?xi64>
      func.call @readDataAsInt(%9, %cast, %cast_1) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, memref<?xmemref<?xi32>>, memref<?xi64>) -> ()
      %12 = memref.get_global @sptprice : memref<1xmemref<?xf64>>
      %cast_2 = memref.cast %12 : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
      %13 = memref.get_global @numOptions : memref<1xi64>
      %cast_3 = memref.cast %13 : memref<1xi64> to memref<?xi64>
      func.call @readDataAsDouble(%9, %cast_2, %cast_3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, memref<?xmemref<?xf64>>, memref<?xi64>) -> ()
      %14 = memref.get_global @strike : memref<1xmemref<?xf64>>
      %cast_4 = memref.cast %14 : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
      %15 = memref.get_global @numOptions : memref<1xi64>
      %cast_5 = memref.cast %15 : memref<1xi64> to memref<?xi64>
      func.call @readDataAsDouble(%9, %cast_4, %cast_5) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, memref<?xmemref<?xf64>>, memref<?xi64>) -> ()
      %16 = memref.get_global @rate : memref<1xmemref<?xf64>>
      %cast_6 = memref.cast %16 : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
      %17 = memref.get_global @numOptions : memref<1xi64>
      %cast_7 = memref.cast %17 : memref<1xi64> to memref<?xi64>
      func.call @readDataAsDouble(%9, %cast_6, %cast_7) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, memref<?xmemref<?xf64>>, memref<?xi64>) -> ()
      %18 = memref.get_global @volatility : memref<1xmemref<?xf64>>
      %cast_8 = memref.cast %18 : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
      %19 = memref.get_global @numOptions : memref<1xi64>
      %cast_9 = memref.cast %19 : memref<1xi64> to memref<?xi64>
      func.call @readDataAsDouble(%9, %cast_8, %cast_9) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, memref<?xmemref<?xf64>>, memref<?xi64>) -> ()
      %20 = memref.get_global @otime : memref<1xmemref<?xf64>>
      %cast_10 = memref.cast %20 : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
      %21 = memref.get_global @numOptions : memref<1xi64>
      %cast_11 = memref.cast %21 : memref<1xi64> to memref<?xi64>
      func.call @readDataAsDouble(%9, %cast_10, %cast_11) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, memref<?xmemref<?xf64>>, memref<?xi64>) -> ()
      %22 = func.call @fclose(%9) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
      %23 = memref.get_global @prices : memref<1xmemref<?xf64>>
      %24 = memref.get_global @numOptions : memref<1xi64>
      %25 = affine.load %24[0] : memref<1xi64>
      %26 = arith.muli %25, %c8_i64 : i64
      %27 = arith.index_cast %26 : i64 to index
      %28 = arith.divui %27, %c8 : index
      %alloc = memref.alloc(%28) : memref<?xf64>
      affine.store %alloc, %23[0] : memref<1xmemref<?xf64>>
      %29 = memref.get_global @prices : memref<1xmemref<?xf64>>
      %30 = affine.load %29[0] : memref<1xmemref<?xf64>>
      %31 = "polygeist.memref2pointer"(%30) : (memref<?xf64>) -> !llvm.ptr
      %32 = llvm.mlir.zero : !llvm.ptr
      %33 = llvm.icmp "eq" %31, %32 : !llvm.ptr
      scf.if %33 {
        %117 = llvm.mlir.addressof @str15 : !llvm.ptr
        %118 = "polygeist.pointer2memref"(%117) : (!llvm.ptr) -> memref<?xi8>
        func.call @die(%118) : (memref<?xi8>) -> ()
      }
      %34 = memref.get_global @prices : memref<1xmemref<?xf64>>
      %35 = affine.load %34[0] : memref<1xmemref<?xf64>>
      %36 = "polygeist.memref2pointer"(%35) : (memref<?xf64>) -> !llvm.ptr
      %37 = memref.get_global @numOptions : memref<1xi64>
      %38 = affine.load %37[0] : memref<1xi64>
      %39 = arith.muli %38, %c8_i64 : i64
      %40 = arith.index_cast %39 : i64 to index
      scf.for %arg2 = %c0 to %40 step %c1 {
        %117 = arith.index_cast %arg2 : index to i32
        %118 = llvm.getelementptr %36[%117] : (!llvm.ptr, i32) -> !llvm.ptr, i8
        llvm.store %c0_i8, %118 : i8, !llvm.ptr
      }
      %41 = func.call @bs_thread() : () -> i32
      %42 = affine.load %alloca_0[] : memref<memref<?xi8>>
      %43 = memref.get_global @prices : memref<1xmemref<?xf64>>
      %44 = affine.load %43[0] : memref<1xmemref<?xf64>>
      %45 = "polygeist.memref2pointer"(%44) : (memref<?xf64>) -> !llvm.ptr
      %46 = "polygeist.pointer2memref"(%45) : (!llvm.ptr) -> memref<?xi8>
      %47 = memref.get_global @numOptions : memref<1xi64>
      %48 = affine.load %47[0] : memref<1xi64>
      func.call @writeQualityFile(%42, %46, %c0_i32, %48) : (memref<?xi8>, memref<?xi8>, i32, i64) -> ()
      %49 = llvm.mlir.addressof @str16 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
      %51 = llvm.call @printf(%50) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %52 = llvm.mlir.addressof @str17 : !llvm.ptr
      %53 = llvm.getelementptr %52[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
      %54 = memref.get_global @numOptions : memref<1xi64>
      %55 = affine.load %54[0] : memref<1xi64>
      %56 = llvm.call @printf(%53, %55) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      %57 = memref.get_global @numOptions : memref<1xi64>
      %58 = affine.load %57[0] : memref<1xi64>
      %59 = arith.cmpi sgt, %58, %c0_i64 : i64
      scf.if %59 {
        %117 = memref.get_global @prices : memref<1xmemref<?xf64>>
        %118 = affine.load %117[0] : memref<1xmemref<?xf64>>
        %119 = affine.load %118[0] : memref<?xf64>
        %120 = memref.get_global @prices : memref<1xmemref<?xf64>>
        %121 = affine.load %120[0] : memref<1xmemref<?xf64>>
        %122 = affine.load %121[0] : memref<?xf64>
        %123 = arith.index_cast %58 : i64 to index
        %124:3 = scf.for %arg2 = %c0 to %123 step %c1 iter_args(%arg3 = %cst, %arg4 = %122, %arg5 = %119) -> (f64, f64, f64) {
          %139 = memref.get_global @prices : memref<1xmemref<?xf64>>
          %140 = affine.load %139[0] : memref<1xmemref<?xf64>>
          %141 = memref.load %140[%arg2] : memref<?xf64>
          %142 = arith.cmpf olt, %141, %arg5 : f64
          %143 = scf.if %142 -> (f64) {
            %147 = memref.load %140[%arg2] : memref<?xf64>
            scf.yield %147 : f64
          } else {
            scf.yield %arg5 : f64
          }
          %144 = arith.cmpf ogt, %141, %arg4 : f64
          %145 = scf.if %144 -> (f64) {
            %147 = memref.load %140[%arg2] : memref<?xf64>
            scf.yield %147 : f64
          } else {
            scf.yield %arg4 : f64
          }
          %146 = arith.addf %arg3, %141 : f64
          scf.yield %146, %145, %143 : f64, f64, f64
        }
        %125 = llvm.mlir.addressof @str18 : !llvm.ptr
        %126 = llvm.getelementptr %125[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<27 x i8>
        %127 = llvm.call @printf(%126) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        %128 = llvm.mlir.addressof @str19 : !llvm.ptr
        %129 = llvm.getelementptr %128[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x i8>
        %130 = arith.sitofp %58 : i64 to f64
        %131 = arith.divf %124#0, %130 : f64
        %132 = llvm.call @printf(%129, %131) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
        %133 = llvm.mlir.addressof @str20 : !llvm.ptr
        %134 = llvm.getelementptr %133[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x i8>
        %135 = llvm.call @printf(%134, %124#2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
        %136 = llvm.mlir.addressof @str21 : !llvm.ptr
        %137 = llvm.getelementptr %136[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x i8>
        %138 = llvm.call @printf(%137, %124#1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      }
      %60 = llvm.mlir.addressof @str22 : !llvm.ptr
      %61 = llvm.getelementptr %60[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<33 x i8>
      %62 = llvm.call @printf(%61) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %63 = llvm.mlir.addressof @str23 : !llvm.ptr
      %64 = llvm.getelementptr %63[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<96 x i8>
      %65 = llvm.call @printf(%64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %66 = llvm.mlir.addressof @str24 : !llvm.ptr
      %67 = llvm.getelementptr %66[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<101 x i8>
      %68 = llvm.call @printf(%67) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %69 = memref.get_global @numOptions : memref<1xi64>
      %70 = affine.load %69[0] : memref<1xi64>
      %71 = arith.cmpi slt, %70, %c20_i64 : i64
      %72 = arith.select %71, %70, %c20_i64 : i64
      %73 = arith.index_cast %72 : i64 to index
      scf.for %arg2 = %c0 to %73 step %c1 {
        %117 = arith.index_cast %arg2 : index to i64
        %118 = llvm.mlir.addressof @str25 : !llvm.ptr
        %119 = llvm.getelementptr %118[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<60 x i8>
        %120 = memref.get_global @sptprice : memref<1xmemref<?xf64>>
        %121 = affine.load %120[0] : memref<1xmemref<?xf64>>
        %122 = memref.load %121[%arg2] : memref<?xf64>
        %123 = memref.get_global @strike : memref<1xmemref<?xf64>>
        %124 = affine.load %123[0] : memref<1xmemref<?xf64>>
        %125 = memref.load %124[%arg2] : memref<?xf64>
        %126 = memref.get_global @rate : memref<1xmemref<?xf64>>
        %127 = affine.load %126[0] : memref<1xmemref<?xf64>>
        %128 = memref.load %127[%arg2] : memref<?xf64>
        %129 = memref.get_global @volatility : memref<1xmemref<?xf64>>
        %130 = affine.load %129[0] : memref<1xmemref<?xf64>>
        %131 = memref.load %130[%arg2] : memref<?xf64>
        %132 = memref.get_global @otime : memref<1xmemref<?xf64>>
        %133 = affine.load %132[0] : memref<1xmemref<?xf64>>
        %134 = memref.load %133[%arg2] : memref<?xf64>
        %135 = memref.get_global @prices : memref<1xmemref<?xf64>>
        %136 = affine.load %135[0] : memref<1xmemref<?xf64>>
        %137 = memref.load %136[%arg2] : memref<?xf64>
        %138 = llvm.call @printf(%119, %117, %122, %125, %128, %131, %134, %137) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64, f64, f64, f64, f64, f64) -> i32
      }
      %74 = llvm.mlir.addressof @str26 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<98 x i8>
      %76 = llvm.call @printf(%75) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %77 = "polygeist.memref2pointer"(%42) : (memref<?xi8>) -> !llvm.ptr
      %78 = func.call @strlen(%77) : (!llvm.ptr) -> i64
      %79 = arith.addi %78, %c5_i64 : i64
      %80 = arith.index_cast %79 : i64 to index
      %alloc_12 = memref.alloc(%80) : memref<?xi8>
      %81 = "polygeist.memref2pointer"(%alloc_12) : (memref<?xi8>) -> !llvm.ptr
      %82 = llvm.mlir.zero : !llvm.ptr
      %83 = llvm.icmp "eq" %81, %82 : !llvm.ptr
      scf.if %83 {
        %117 = llvm.mlir.addressof @str27 : !llvm.ptr
        %118 = "polygeist.pointer2memref"(%117) : (!llvm.ptr) -> memref<?xi8>
        func.call @die(%118) : (memref<?xi8>) -> ()
      }
      %84 = arith.index_cast %78 : i64 to index
      scf.for %arg2 = %c0 to %84 step %c1 {
        %117 = memref.load %42[%arg2] : memref<?xi8>
        memref.store %117, %alloc_12[%arg2] : memref<?xi8>
      }
      %85 = arith.index_cast %78 : i64 to index
      %86 = llvm.mlir.addressof @str28 : !llvm.ptr
      scf.for %arg2 = %c0 to %c5 step %c1 {
        %117 = arith.index_cast %arg2 : index to i32
        %118 = llvm.getelementptr %86[%117] : (!llvm.ptr, i32) -> !llvm.ptr, i8
        %119 = llvm.load %118 : !llvm.ptr -> i8
        %120 = arith.addi %arg2, %85 : index
        memref.store %119, %alloc_12[%120] : memref<?xi8>
      }
      %87 = memref.get_global @otype : memref<1xmemref<?xi32>>
      %88 = affine.load %87[0] : memref<1xmemref<?xi32>>
      %89 = memref.get_global @sptprice : memref<1xmemref<?xf64>>
      %90 = affine.load %89[0] : memref<1xmemref<?xf64>>
      %91 = memref.get_global @strike : memref<1xmemref<?xf64>>
      %92 = affine.load %91[0] : memref<1xmemref<?xf64>>
      %93 = memref.get_global @rate : memref<1xmemref<?xf64>>
      %94 = affine.load %93[0] : memref<1xmemref<?xf64>>
      %95 = memref.get_global @volatility : memref<1xmemref<?xf64>>
      %96 = affine.load %95[0] : memref<1xmemref<?xf64>>
      %97 = memref.get_global @otime : memref<1xmemref<?xf64>>
      %98 = affine.load %97[0] : memref<1xmemref<?xf64>>
      %99 = memref.get_global @prices : memref<1xmemref<?xf64>>
      %100 = affine.load %99[0] : memref<1xmemref<?xf64>>
      %101 = memref.get_global @numOptions : memref<1xi64>
      %102 = affine.load %101[0] : memref<1xi64>
      func.call @writeTextResults(%alloc_12, %88, %90, %92, %94, %96, %98, %100, %102) : (memref<?xi8>, memref<?xi32>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, i64) -> ()
      memref.dealloc %alloc_12 : memref<?xi8>
      %103 = memref.get_global @sptprice : memref<1xmemref<?xf64>>
      %104 = affine.load %103[0] : memref<1xmemref<?xf64>>
      memref.dealloc %104 : memref<?xf64>
      %105 = memref.get_global @strike : memref<1xmemref<?xf64>>
      %106 = affine.load %105[0] : memref<1xmemref<?xf64>>
      memref.dealloc %106 : memref<?xf64>
      %107 = memref.get_global @rate : memref<1xmemref<?xf64>>
      %108 = affine.load %107[0] : memref<1xmemref<?xf64>>
      memref.dealloc %108 : memref<?xf64>
      %109 = memref.get_global @volatility : memref<1xmemref<?xf64>>
      %110 = affine.load %109[0] : memref<1xmemref<?xf64>>
      memref.dealloc %110 : memref<?xf64>
      %111 = memref.get_global @otime : memref<1xmemref<?xf64>>
      %112 = affine.load %111[0] : memref<1xmemref<?xf64>>
      memref.dealloc %112 : memref<?xf64>
      %113 = memref.get_global @otype : memref<1xmemref<?xi32>>
      %114 = affine.load %113[0] : memref<1xmemref<?xi32>>
      memref.dealloc %114 : memref<?xi32>
      %115 = memref.get_global @prices : memref<1xmemref<?xf64>>
      %116 = affine.load %115[0] : memref<1xmemref<?xf64>>
      memref.dealloc %116 : memref<?xf64>
    }
    return %8 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strlen(!llvm.ptr) -> i64
  func.func private @writeTextResults(%arg0: memref<?xi8>, %arg1: memref<?xi32>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: i64) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = llvm.mlir.addressof @str30 : !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi8>
    %2 = call @fopen(%arg0, %1) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %3 = "polygeist.memref2pointer"(%2) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.icmp "eq" %3, %4 : !llvm.ptr
    scf.if %5 {
      %10 = llvm.mlir.addressof @str31 : !llvm.ptr
      %11 = "polygeist.pointer2memref"(%10) : (!llvm.ptr) -> memref<?xi8>
      func.call @die(%11) : (memref<?xi8>) -> ()
    }
    %6 = llvm.mlir.addressof @str32 : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<57 x i8>
    %8 = llvm.call @fprintf(%3, %7) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    call @dump_text_rows(%2, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, memref<?xi32>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, i64) -> ()
    %9 = call @fclose(%2) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
    return
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @decide_bs_state(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 2.000000e-01 : f64
    %cst_0 = arith.constant 3.000000e-01 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %0 = arith.divf %arg0, %arg1 : f64
    %1 = arith.subf %0, %cst_1 : f64
    %2 = math.absf %1 : f64
    %3 = math.absf %arg2 : f64
    %4 = arith.mulf %3, %arg3 : f64
    %5 = arith.cmpf ogt, %2, %cst_0 : f64
    %6 = arith.extui %5 : i1 to i32
    %7 = arith.cmpf ogt, %4, %cst : f64
    %8 = arith.extui %7 : i1 to i32
    %9 = arith.addi %6, %8 : i32
    return %9 : i32
  }
  func.func private @BlkSchlsEqEuroNoDiv(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64, %arg5: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 1.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %0 = math.sqrt %arg4 : f64
    %1 = arith.divf %arg0, %arg1 : f64
    %2 = math.log %1 : f64
    %3 = arith.mulf %arg3, %arg3 : f64
    %4 = memref.get_global @half : memref<1xf64>
    %5 = affine.load %4[0] : memref<1xf64>
    %6 = arith.mulf %3, %5 : f64
    %7 = arith.addf %arg2, %6 : f64
    %8 = arith.mulf %7, %arg4 : f64
    %9 = arith.addf %8, %2 : f64
    %10 = arith.mulf %arg3, %0 : f64
    %11 = arith.divf %9, %10 : f64
    %12 = arith.mulf %arg3, %0 : f64
    %13 = arith.subf %11, %12 : f64
    %14 = call @compute_cndf(%11, %c0_i32) : (f64, i32) -> f64
    %15 = call @compute_cndf(%13, %c0_i32) : (f64, i32) -> f64
    %16 = arith.negf %arg2 : f64
    %17 = arith.mulf %16, %arg4 : f64
    %18 = math.exp %17 : f64
    %19 = arith.mulf %18, %arg1 : f64
    %20 = arith.subf %cst, %15 : f64
    %21 = arith.mulf %19, %20 : f64
    %22 = arith.subf %cst, %14 : f64
    %23 = arith.mulf %arg0, %22 : f64
    %24 = arith.subf %21, %23 : f64
    return %24 : f64
  }
  func.func private @dump_text_rows(%arg0: memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>, %arg1: memref<?xi32>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: i64) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg8 : i64 to index
    scf.for %arg9 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg9 : index to i64
      %2 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %3 = llvm.mlir.addressof @str33 : !llvm.ptr
      %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<44 x i8>
      %5 = memref.load %arg1[%arg9] : memref<?xi32>
      %6 = memref.load %arg2[%arg9] : memref<?xf64>
      %7 = memref.load %arg3[%arg9] : memref<?xf64>
      %8 = memref.load %arg4[%arg9] : memref<?xf64>
      %9 = memref.load %arg5[%arg9] : memref<?xf64>
      %10 = memref.load %arg6[%arg9] : memref<?xf64>
      %11 = memref.load %arg7[%arg9] : memref<?xf64>
      %12 = llvm.call @fprintf(%2, %4, %1, %5, %6, %7, %8, %9, %10, %11) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, i64, i32, f64, f64, f64, f64, f64, f64) -> i32
    }
    return
  }
}
