module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  // Knob for model_choose function, enabling task skipping.
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "model_choose",
    transform_type = "task_skipping",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 4>,
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1, 2>, // Corresponds to different models/tasks
    thresholds = array<i32: 3>,
    decisions = array<i32: 2, 1>
  }> : () -> ()
  llvm.mlir.global internal constant @str1("Based on the input and state, the chosen model is: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Usage: %s <state>\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @model_choose(%arg0: i32, %arg1: memref<?xi32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c20_i32 = arith.constant 20 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %0 = arith.cmpi sgt, %arg0, %c10_i32 : i32
    scf.if %0 {
      affine.store %c1_i32, %arg1[0] : memref<?xi32>
    } else {
      %1 = arith.cmpi sgt, %arg0, %c20_i32 : i32
      scf.if %1 {
        affine.store %c2_i32, %arg1[0] : memref<?xi32>
      }
    }
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %alloca = memref.alloca() : memref<1xi32>
    %0 = llvm.mlir.undef : i32
    affine.store %0, %alloca[0] : memref<1xi32>
    %1 = arith.cmpi ne, %arg0, %c2_i32 : i32
    %2 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %3 = arith.select %1, %c1_i32, %0 : i32
    scf.if %1 {
      %5 = llvm.mlir.addressof @str0 : !llvm.ptr
      %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
      %7 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %8 = "polygeist.memref2pointer"(%7) : (memref<?xi8>) -> !llvm.ptr
      %9 = llvm.call @printf(%6, %8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    %4 = arith.select %2, %c0_i32, %3 : i32
    scf.if %2 {
      %5 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %6 = func.call @atoi(%5) : (memref<?xi8>) -> i32
      %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
      func.call @model_choose(%c15_i32, %cast, %6) : (i32, memref<?xi32>, i32) -> ()
      %7 = llvm.mlir.addressof @str1 : !llvm.ptr
      %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<55 x i8>
      %9 = affine.load %alloca[0] : memref<1xi32>
      %10 = llvm.call @printf(%8, %9) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    }
    return %4 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}
