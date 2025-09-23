module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str59("  query string : BM25 query (default \22quick brown fox\22)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str58("  seed         : RNG seed (default 42)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str57("  num_docs     : number of documents to generate (default 6)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str56("Usage: %s <num_docs> [seed] [\22query string\22]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str55("\0AComputation time: %.3f ms\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str54("An error occurred during ranking.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str53("Rank %d: Doc %d (Score: %.4f) - \22%s\22\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str52("generate_corpus failed\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str51("Ranking documents:\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str50("Query: \22%s\22\0A\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str49("num_docs must be positive.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str48(" %s\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str47("%s\00") {addr_space = 0 : i32}
  llvm.func @snprintf(!llvm.ptr, i64, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str46("measure\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str45("distance\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str44("approximate\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str43("modeling\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str42("language\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str41("relevance\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str40("quality\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str39("pagerank\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str38("walk\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str37("random\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str36("graph\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str35("embedding\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str34("analysis\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str33("weight\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str32("field\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str31("length\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str30("normalize\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str29("tfidf\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str28("space\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str27("vector\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str26("index\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str25("token\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str24("score\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str23("probabilistic\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str22("inverse\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str21("frequency\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str20("term\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str19("model\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str18("query\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("document\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("text\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("system\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("information\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("bm25\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("ranking\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("retrieval\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("engine\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("search\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("dog\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("lazy\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("over\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("jumps\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("fox\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("brown\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("quick\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("the\00") {addr_space = 0 : i32}
  memref.global "private" @"generate_corpus@static@vocab@init" : memref<1xi1> = dense<true>
  memref.global "private" @"generate_corpus@static@vocab" : memref<47xmemref<?xi8>> = uninitialized
  llvm.mlir.global internal constant @str0(" .,;:!?\22'\0A\09()[]{}<>\00") {addr_space = 0 : i32}
  memref.global @B : memref<1xf64> = dense<7.500000e-01>
  memref.global @K1 : memref<1xf64> = dense<1.500000e+00>
  func.func @compare_tokens(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = call @strcasecmp(%arg0, %arg1) : (memref<?xi8>, memref<?xi8>) -> i32
    return %0 : i32
  }
  func.func private @strcasecmp(memref<?xi8>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @count_and_lower_words(%arg0: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i8 = arith.constant 0 : i8
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
      %5:3 = scf.while (%arg1 = %c0_i32, %arg2 = %c0_i32, %arg3 = %arg0) : (i32, i32, memref<?xi8>) -> (i32, i32, memref<?xi8>) {
        %6 = affine.load %arg3[0] : memref<?xi8>
        %7 = arith.cmpi ne, %6, %c0_i8 : i8
        scf.condition(%7) %arg2, %arg1, %arg3 : i32, i32, memref<?xi8>
      } do {
      ^bb0(%arg1: i32, %arg2: i32, %arg3: memref<?xi8>):
        %6 = affine.load %arg3[0] : memref<?xi8>
        %7 = arith.extui %6 : i8 to i32
        %8 = func.call @tolower(%7) : (i32) -> i32
        %9 = arith.trunci %8 : i32 to i8
        affine.store %9, %arg3[0] : memref<?xi8>
        %10 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
        %11 = affine.load %10[0] : memref<?xmemref<?xi16>>
        %12 = affine.load %arg3[0] : memref<?xi8>
        %13 = arith.extui %12 : i8 to i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = memref.load %11[%14] : memref<?xi16>
        %16 = arith.extui %15 : i16 to i32
        %17 = arith.andi %16, %c8_i32 : i32
        %18 = arith.cmpi ne, %17, %c0_i32 : i32
        %19 = arith.cmpi eq, %arg2, %c0_i32 : i32
        %20 = arith.select %19, %c1_i32, %arg2 : i32
        %21 = arith.andi %18, %19 : i1
        %22 = arith.select %18, %20, %c0_i32 : i32
        %23 = scf.if %21 -> (i32) {
          %25 = arith.addi %arg1, %c1_i32 : i32
          scf.yield %25 : i32
        } else {
          scf.yield %arg1 : i32
        }
        %24 = "polygeist.subindex"(%arg3, %c1) : (memref<?xi8>, index) -> memref<?xi8>
        scf.yield %22, %23, %24 : i32, i32, memref<?xi8>
      }
      scf.yield %5#0 : i32
    }
    return %4 : i32
  }
  func.func private @tolower(i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__ctype_b_loc() -> memref<?xmemref<?xi16>> attributes {llvm.linkage = #llvm.linkage<external>}
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
  func.func @__internal_tf_count_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
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
        %14:2 = scf.while (%arg3 = %c0_i32, %arg4 = %arg1) : (i32, memref<?xi8>) -> (i32, memref<?xi8>) {
          %15 = func.call @strstr(%arg4, %arg0) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
          %16 = "polygeist.memref2pointer"(%15) : (memref<?xi8>) -> !llvm.ptr
          %17 = llvm.icmp "ne" %16, %11 : !llvm.ptr
          scf.condition(%17) %arg3, %15 : i32, memref<?xi8>
        } do {
        ^bb0(%arg3: i32, %arg4: memref<?xi8>):
          %15 = "polygeist.memref2pointer"(%arg4) : (memref<?xi8>) -> !llvm.ptr
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
          %18 = memref.load %arg4[%13] : memref<?xi8>
          %19 = arith.extsi %18 : i8 to i32
          %20 = arith.cmpi eq, %19, %c0_i32 : i32
          %21 = scf.if %20 -> (i1) {
            scf.yield %true : i1
          } else {
            %25 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %26 = affine.load %25[0] : memref<?xmemref<?xi16>>
            %27 = memref.load %arg4[%13] : memref<?xi8>
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
            %25 = arith.addi %arg3, %c1_i32 : i32
            scf.yield %25 : i32
          } else {
            scf.yield %arg3 : i32
          }
          %24 = "polygeist.subindex"(%arg4, %13) : (memref<?xi8>, index) -> memref<?xi8>
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
  func.func @tf_count_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sge, %arg2, %c5_i32 : i32
    %1 = arith.select %0, %c1_i32, %c0_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = scf.index_switch %2 -> i32 
    case 0 {
      %4 = func.call @__internal_tf_count_whole_word(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xi8>, i32) -> i32
      scf.yield %4 : i32
    }
    case 1 {
      %4 = func.call @approx_tf_count_whole_word(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xi8>, i32) -> i32
      scf.yield %4 : i32
    }
    default {
      %4 = func.call @__internal_tf_count_whole_word(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xi8>, i32) -> i32
      scf.yield %4 : i32
    }
    return %3 : i32
  }
  func.func private @strlen(!llvm.ptr) -> i64
  func.func private @strstr(memref<?xi8>, memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @approx_tf_count_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
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
        %12 = arith.index_cast %8 : i64 to index
        %13:2 = scf.while (%arg3 = %c0_i32, %arg4 = %arg1) : (i32, memref<?xi8>) -> (i32, memref<?xi8>) {
          %14 = func.call @strstr(%arg4, %arg0) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
          %15 = "polygeist.memref2pointer"(%14) : (memref<?xi8>) -> !llvm.ptr
          %16 = llvm.icmp "ne" %15, %11 : !llvm.ptr
          scf.condition(%16) %arg3, %14 : i32, memref<?xi8>
        } do {
        ^bb0(%arg3: i32, %arg4: memref<?xi8>):
          %14 = arith.addi %arg3, %c1_i32 : i32
          %15 = "polygeist.subindex"(%arg4, %12) : (memref<?xi8>, index) -> memref<?xi8>
          scf.yield %14, %15 : i32, memref<?xi8>
        }
        scf.yield %13#0 : i32
      } else {
        scf.yield %5 : i32
      }
      scf.yield %10 : i32
    }
    return %6 : i32
  }
  func.func @__internal_df_contains_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
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
        %15:3 = scf.while (%arg3 = %true, %arg4 = %13, %arg5 = %true, %arg6 = %arg1) : (i1, i32, i1, memref<?xi8>) -> (i1, i32, memref<?xi8>) {
          %16 = func.call @strstr(%arg6, %arg0) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
          %17 = "polygeist.memref2pointer"(%16) : (memref<?xi8>) -> !llvm.ptr
          %18 = llvm.icmp "ne" %17, %14 : !llvm.ptr
          %19 = arith.andi %18, %arg5 : i1
          scf.condition(%19) %arg3, %arg4, %16 : i1, i32, memref<?xi8>
        } do {
        ^bb0(%arg3: i1, %arg4: i32, %arg5: memref<?xi8>):
          %16 = "polygeist.memref2pointer"(%arg5) : (memref<?xi8>) -> !llvm.ptr
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
          %23 = memref.load %arg5[%22] : memref<?xi8>
          %24 = arith.extsi %23 : i8 to i32
          %25 = arith.cmpi eq, %24, %c0_i32 : i32
          %26 = scf.if %25 -> (i1) {
            scf.yield %true : i1
          } else {
            %33 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %34 = affine.load %33[0] : memref<?xmemref<?xi16>>
            %35 = memref.load %arg5[%22] : memref<?xi8>
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
            %34 = arith.andi %33, %arg3 : i1
            %35 = arith.select %26, %c1_i32, %arg4 : i32
            scf.yield %34, %35 : i1, i32
          } else {
            scf.yield %arg3, %arg4 : i1, i32
          }
          %32 = scf.if %30 -> (memref<?xi8>) {
            %33 = arith.index_cast %9 : i64 to index
            %34 = "polygeist.subindex"(%arg5, %33) : (memref<?xi8>, index) -> memref<?xi8>
            scf.yield %34 : memref<?xi8>
          } else {
            scf.yield %arg5 : memref<?xi8>
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
  func.func @df_contains_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sge, %arg2, %c3_i32 : i32
    %1 = arith.select %0, %c1_i32, %c0_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = scf.index_switch %2 -> i32 
    case 0 {
      %4 = func.call @__internal_df_contains_whole_word(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xi8>, i32) -> i32
      scf.yield %4 : i32
    }
    case 1 {
      %4 = func.call @approx_df_contains_whole_word(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xi8>, i32) -> i32
      scf.yield %4 : i32
    }
    default {
      %4 = func.call @__internal_df_contains_whole_word(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xi8>, i32) -> i32
      scf.yield %4 : i32
    }
    return %3 : i32
  }
  func.func @approx_df_contains_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.icmp "eq" %0, %1 : !llvm.ptr
    %3 = scf.if %2 -> (i1) {
      scf.yield %true : i1
    } else {
      %5 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
      %6 = llvm.icmp "eq" %5, %1 : !llvm.ptr
      scf.yield %6 : i1
    }
    %4 = scf.if %3 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %5 = func.call @strstr(%arg1, %arg0) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
      %6 = "polygeist.memref2pointer"(%5) : (memref<?xi8>) -> !llvm.ptr
      %7 = llvm.mlir.zero : !llvm.ptr
      %8 = llvm.icmp "ne" %6, %7 : !llvm.ptr
      %9 = arith.extui %8 : i1 to i32
      scf.yield %9 : i32
    }
    return %4 : i32
  }
  func.func @calculate_df(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
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
        %12 = "polygeist.memref2pointer"(%7) : (memref<?xi8>) -> !llvm.ptr
        %13 = func.call @strlen(%12) : (!llvm.ptr) -> i64
        %14 = arith.trunci %13 : i64 to i32
        %15 = arith.cmpi sgt, %14, %c16_i32 : i32
        %16 = arith.select %15, %c16_i32, %14 : i32
        %17 = arith.index_cast %arg2 : i32 to index
        %18 = scf.for %arg3 = %c0 to %17 step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
          %19 = memref.load %arg1[%arg3] : memref<?xmemref<?xi8>>
          %20 = func.call @strdup(%19) : (memref<?xi8>) -> memref<?xi8>
          %21 = "polygeist.memref2pointer"(%20) : (memref<?xi8>) -> !llvm.ptr
          %22 = llvm.mlir.zero : !llvm.ptr
          %23 = llvm.icmp "eq" %21, %22 : !llvm.ptr
          %24 = scf.if %23 -> (i32) {
            scf.yield %arg4 : i32
          } else {
            %25 = scf.while (%arg5 = %20) : (memref<?xi8>) -> memref<?xi8> {
              %29 = affine.load %arg5[0] : memref<?xi8>
              %30 = arith.cmpi ne, %29, %c0_i8 : i8
              scf.condition(%30) %arg5 : memref<?xi8>
            } do {
            ^bb0(%arg5: memref<?xi8>):
              %29 = affine.load %arg5[0] : memref<?xi8>
              %30 = arith.extui %29 : i8 to i32
              %31 = func.call @tolower(%30) : (i32) -> i32
              %32 = arith.trunci %31 : i32 to i8
              affine.store %32, %arg5[0] : memref<?xi8>
              %33 = "polygeist.subindex"(%arg5, %c1) : (memref<?xi8>, index) -> memref<?xi8>
              scf.yield %33 : memref<?xi8>
            }
            %26 = func.call @df_contains_whole_word(%7, %20, %16) : (memref<?xi8>, memref<?xi8>, i32) -> i32
            %27 = arith.cmpi ne, %26, %c0_i32 : i32
            %28 = scf.if %27 -> (i32) {
              %29 = arith.addi %arg4, %c1_i32 : i32
              scf.yield %29 : i32
            } else {
              scf.yield %arg4 : i32
            }
            memref.dealloc %20 : memref<?xi8>
            scf.yield %28 : i32
          }
          scf.yield %24 : i32
        }
        memref.dealloc %7 : memref<?xi8>
        scf.yield %18 : i32
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
  func.func private @strdup(memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @score_term_over_docs(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xf64>, %arg3: f64, %arg4: f64, %arg5: memref<?x!llvm.struct<(i32, f64)>>, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c113_i32 = arith.constant 113 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.cmpi sge, %arg7, %c113_i32 : i32
    %1 = arith.select %0, %c1_i32, %c0_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    scf.index_switch %2 
    case 0 {
      %3 = arith.index_cast %arg6 : i32 to index
      scf.for %arg8 = %c0 to %3 step %c1 {
        %4 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
        %5 = func.call @strlen(%4) : (!llvm.ptr) -> i64
        %6 = arith.trunci %5 : i64 to i32
        %7 = arith.cmpi sgt, %6, %c16_i32 : i32
        %8 = arith.select %7, %c16_i32, %6 : i32
        %9 = memref.load %arg1[%arg8] : memref<?xmemref<?xi8>>
        %10 = func.call @tf_count_whole_word(%arg0, %9, %8) : (memref<?xi8>, memref<?xi8>, i32) -> i32
        %11 = arith.sitofp %10 : i32 to f64
        %12 = memref.get_global @K1 : memref<1xf64>
        %13 = affine.load %12[0] : memref<1xf64>
        %14 = arith.addf %13, %cst : f64
        %15 = arith.mulf %11, %14 : f64
        %16 = memref.get_global @B : memref<1xf64>
        %17 = affine.load %16[0] : memref<1xf64>
        %18 = arith.subf %cst, %17 : f64
        %19 = memref.load %arg2[%arg8] : memref<?xf64>
        %20 = arith.divf %19, %arg3 : f64
        %21 = arith.mulf %17, %20 : f64
        %22 = arith.addf %18, %21 : f64
        %23 = arith.mulf %13, %22 : f64
        %24 = arith.addf %11, %23 : f64
        %25 = arith.cmpf ogt, %24, %cst_0 : f64
        %26 = scf.if %25 -> (f64) {
          %35 = arith.divf %15, %24 : f64
          scf.yield %35 : f64
        } else {
          scf.yield %cst_0 : f64
        }
        %27 = arith.mulf %arg4, %26 : f64
        %28 = arith.muli %arg8, %c16 : index
        %29 = arith.index_cast %28 : index to i64
        %30 = "polygeist.memref2pointer"(%arg5) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %31 = llvm.getelementptr %30[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %32 = llvm.getelementptr %31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        %33 = llvm.load %32 : !llvm.ptr -> f64
        %34 = arith.addf %33, %27 : f64
        llvm.store %34, %32 : f64, !llvm.ptr
      }
      scf.yield
    }
    case 1 {
      %3 = arith.index_cast %arg6 : i32 to index
      scf.for %arg8 = %c0 to %3 step %c1 {
        %4 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
        %5 = func.call @strlen(%4) : (!llvm.ptr) -> i64
        %6 = arith.trunci %5 : i64 to i32
        %7 = arith.cmpi sgt, %6, %c16_i32 : i32
        %8 = arith.select %7, %c16_i32, %6 : i32
        %9 = memref.load %arg1[%arg8] : memref<?xmemref<?xi8>>
        %10 = func.call @tf_count_whole_word(%arg0, %9, %8) : (memref<?xi8>, memref<?xi8>, i32) -> i32
        %11 = arith.sitofp %10 : i32 to f64
        %12 = memref.get_global @K1 : memref<1xf64>
        %13 = affine.load %12[0] : memref<1xf64>
        %14 = arith.addf %13, %cst : f64
        %15 = arith.mulf %11, %14 : f64
        %16 = memref.get_global @B : memref<1xf64>
        %17 = affine.load %16[0] : memref<1xf64>
        %18 = arith.subf %cst, %17 : f64
        %19 = memref.load %arg2[%arg8] : memref<?xf64>
        %20 = arith.divf %19, %arg3 : f64
        %21 = arith.mulf %17, %20 : f64
        %22 = arith.addf %18, %21 : f64
        %23 = arith.mulf %13, %22 : f64
        %24 = arith.addf %11, %23 : f64
        %25 = arith.cmpf ogt, %24, %cst_0 : f64
        %26 = scf.if %25 -> (f64) {
          %35 = arith.divf %15, %24 : f64
          scf.yield %35 : f64
        } else {
          scf.yield %cst_0 : f64
        }
        %27 = arith.mulf %arg4, %26 : f64
        %28 = arith.muli %arg8, %c16 : index
        %29 = arith.index_cast %28 : index to i64
        %30 = "polygeist.memref2pointer"(%arg5) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %31 = llvm.getelementptr %30[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %32 = llvm.getelementptr %31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        %33 = llvm.load %32 : !llvm.ptr -> f64
        %34 = arith.addf %33, %27 : f64
        llvm.store %34, %32 : f64, !llvm.ptr
      }
      scf.yield
    }
    default {
      %3 = arith.index_cast %arg6 : i32 to index
      scf.for %arg8 = %c0 to %3 step %c1 {
        %4 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
        %5 = func.call @strlen(%4) : (!llvm.ptr) -> i64
        %6 = arith.trunci %5 : i64 to i32
        %7 = arith.cmpi sgt, %6, %c16_i32 : i32
        %8 = arith.select %7, %c16_i32, %6 : i32
        %9 = memref.load %arg1[%arg8] : memref<?xmemref<?xi8>>
        %10 = func.call @tf_count_whole_word(%arg0, %9, %8) : (memref<?xi8>, memref<?xi8>, i32) -> i32
        %11 = arith.sitofp %10 : i32 to f64
        %12 = memref.get_global @K1 : memref<1xf64>
        %13 = affine.load %12[0] : memref<1xf64>
        %14 = arith.addf %13, %cst : f64
        %15 = arith.mulf %11, %14 : f64
        %16 = memref.get_global @B : memref<1xf64>
        %17 = affine.load %16[0] : memref<1xf64>
        %18 = arith.subf %cst, %17 : f64
        %19 = memref.load %arg2[%arg8] : memref<?xf64>
        %20 = arith.divf %19, %arg3 : f64
        %21 = arith.mulf %17, %20 : f64
        %22 = arith.addf %18, %21 : f64
        %23 = arith.mulf %13, %22 : f64
        %24 = arith.addf %11, %23 : f64
        %25 = arith.cmpf ogt, %24, %cst_0 : f64
        %26 = scf.if %25 -> (f64) {
          %35 = arith.divf %15, %24 : f64
          scf.yield %35 : f64
        } else {
          scf.yield %cst_0 : f64
        }
        %27 = arith.mulf %arg4, %26 : f64
        %28 = arith.muli %arg8, %c16 : index
        %29 = arith.index_cast %28 : index to i64
        %30 = "polygeist.memref2pointer"(%arg5) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %31 = llvm.getelementptr %30[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %32 = llvm.getelementptr %31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        %33 = llvm.load %32 : !llvm.ptr -> f64
        %34 = arith.addf %33, %27 : f64
        llvm.store %34, %32 : f64, !llvm.ptr
      }
    }
    return
  }
  func.func @rank_documents_bm25(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: i32) -> memref<?x!llvm.struct<(i32, f64)>> attributes {llvm.linkage = #llvm.linkage<external>} {
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
        %22:4 = scf.while (%arg3 = %0, %arg4 = %c0_i32, %arg5 = %cst, %arg6 = %true, %arg7 = %true) : (i32, i32, f64, i1, i1) -> (f64, i1, i32, i32) {
          %23 = arith.cmpi slt, %arg4, %arg2 : i32
          %24 = arith.andi %23, %arg7 : i1
          scf.condition(%24) %arg5, %arg6, %arg3, %arg4 : f64, i1, i32, i32
        } do {
        ^bb0(%arg3: f64, %arg4: i1, %arg5: i32, %arg6: i32):
          %23 = arith.index_cast %arg6 : i32 to index
          %24 = memref.load %arg1[%23] : memref<?xmemref<?xi8>>
          %25 = func.call @strdup(%24) : (memref<?xi8>) -> memref<?xi8>
          %26 = "polygeist.memref2pointer"(%25) : (memref<?xi8>) -> !llvm.ptr
          %27 = llvm.mlir.zero : !llvm.ptr
          %28 = llvm.icmp "eq" %26, %27 : !llvm.ptr
          %29 = arith.xori %28, %true : i1
          %30 = arith.andi %29, %arg4 : i1
          %31 = arith.xori %28, %true : i1
          %32 = arith.select %28, %arg6, %arg5 : i32
          %33:2 = scf.if %28 -> (f64, i32) {
            %34 = arith.index_cast %arg6 : i32 to index
            scf.for %arg7 = %c0 to %34 step %c1 {
              %36 = memref.load %alloc_2[%arg7] : memref<?xmemref<?xi8>>
              memref.dealloc %36 : memref<?xi8>
            }
            memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
            memref.dealloc %alloc : memref<?xf64>
            %35 = "polygeist.pointer2memref"(%27) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
            affine.store %35, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
            scf.yield %arg3, %arg6 : f64, i32
          } else {
            %34 = arith.index_cast %arg6 : i32 to index
            %35 = func.call @count_and_lower_words(%25) : (memref<?xi8>) -> i32
            %36 = arith.sitofp %35 : i32 to f64
            memref.store %36, %alloc[%34] : memref<?xf64>
            %37 = arith.index_cast %arg6 : i32 to index
            %38 = memref.load %alloc[%37] : memref<?xf64>
            %39 = arith.addf %arg3, %38 : f64
            %40 = arith.index_cast %arg6 : i32 to index
            memref.store %25, %alloc_2[%40] : memref<?xmemref<?xi8>>
            %41 = arith.addi %arg6, %c1_i32 : i32
            scf.yield %39, %41 : f64, i32
          }
          scf.yield %32, %33#1, %33#0, %30, %31 : i32, i32, f64, i1, i1
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
            scf.for %arg3 = %c0 to %32 step %c1 {
              %34 = memref.load %alloc_2[%arg3] : memref<?xmemref<?xi8>>
              memref.dealloc %34 : memref<?xi8>
            }
            memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
            memref.dealloc %alloc : memref<?xf64>
            %33 = "polygeist.pointer2memref"(%30) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
            affine.store %33, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
          } else {
            %32 = arith.index_cast %arg2 : i32 to index
            scf.for %arg3 = %c0 to %32 step %c1 {
              %37 = arith.index_cast %arg3 : index to i32
              %38 = arith.muli %arg3, %c16 : index
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
              scf.for %arg3 = %c0 to %37 step %c1 {
                %39 = memref.load %alloc_2[%arg3] : memref<?xmemref<?xi8>>
                memref.dealloc %39 : memref<?xi8>
              }
              memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
              memref.dealloc %alloc : memref<?xf64>
              memref.dealloc %alloc_3 : memref<?x!llvm.struct<(i32, f64)>>
              %38 = "polygeist.pointer2memref"(%35) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
              affine.store %38, %alloca_1[] : memref<memref<?x!llvm.struct<(i32, f64)>>>
            } else {
              affine.store %33, %alloca[0] : memref<1xmemref<?xi8>>
              %37 = llvm.mlir.addressof @str0 : !llvm.ptr
              %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
              %38 = "polygeist.pointer2memref"(%37) : (!llvm.ptr) -> memref<?xi8>
              %39 = llvm.mlir.zero : !llvm.ptr
              %40:5 = scf.while (%arg3 = %1, %arg4 = %0, %arg5 = %0, %arg6 = %0, %arg7 = %c0_i32, %arg8 = %true) : (f64, i32, i32, i32, i32, i1) -> (i32, f64, i32, i32, i32) {
                scf.condition(%arg8) %arg7, %arg3, %arg4, %arg5, %arg6 : i32, f64, i32, i32, i32
              } do {
              ^bb0(%arg3: i32, %arg4: f64, %arg5: i32, %arg6: i32, %arg7: i32):
                %43 = affine.load %alloca[0] : memref<1xmemref<?xi8>>
                %44 = func.call @strtok_r(%43, %38, %cast) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
                %45 = "polygeist.memref2pointer"(%44) : (memref<?xi8>) -> !llvm.ptr
                %46 = llvm.icmp "eq" %45, %39 : !llvm.ptr
                %47:6 = scf.if %46 -> (i32, i32, i1, i32, i32, f64) {
                  scf.yield %arg6, %arg7, %false, %arg3, %arg5, %arg4 : i32, i32, i1, i32, i32, f64
                } else {
                  %48 = affine.load %44[0] : memref<?xi8>
                  %49 = arith.extsi %48 : i8 to i32
                  %50 = arith.cmpi ne, %49, %c0_i32 : i32
                  %51:6 = scf.if %50 -> (i32, i32, i1, i32, i32, f64) {
                    %52:2 = scf.while (%arg8 = %c0_i32, %arg9 = %c0_i32, %arg10 = %true) : (i32, i32, i1) -> (i32, i32) {
                      %61 = arith.cmpi slt, %arg8, %arg3 : i32
                      %62 = arith.andi %61, %arg10 : i1
                      scf.condition(%62) %arg8, %arg9 : i32, i32
                    } do {
                    ^bb0(%arg8: i32, %arg9: i32):
                      %61 = arith.index_cast %arg8 : i32 to index
                      %62 = memref.load %alloca_0[%61] : memref<256xmemref<?xi8>>
                      %63 = func.call @compare_tokens(%44, %62) : (memref<?xi8>, memref<?xi8>) -> i32
                      %64 = arith.cmpi eq, %63, %c0_i32 : i32
                      %65 = arith.select %64, %c1_i32, %arg9 : i32
                      %66 = arith.cmpi ne, %63, %c0_i32 : i32
                      %67 = scf.if %66 -> (i32) {
                        %68 = arith.addi %arg8, %c1_i32 : i32
                        scf.yield %68 : i32
                      } else {
                        scf.yield %arg8 : i32
                      }
                      scf.yield %67, %65, %66 : i32, i32, i1
                    }
                    %53 = arith.cmpi eq, %52#1, %c0_i32 : i32
                    %54 = arith.cmpi slt, %arg3, %c256_i32 : i32
                    %55 = arith.cmpi slt, %arg3, %c256_i32 : i32
                    %56 = arith.andi %53, %54 : i1
                    %57 = arith.cmpi ne, %52#1, %c0_i32 : i32
                    %58 = arith.andi %53, %55 : i1
                    %59 = arith.ori %58, %57 : i1
                    %60:3 = scf.if %56 -> (i32, i32, f64) {
                      %61 = arith.index_cast %arg3 : i32 to index
                      %62 = func.call @strdup(%44) : (memref<?xi8>) -> memref<?xi8>
                      memref.store %62, %alloca_0[%61] : memref<256xmemref<?xi8>>
                      %63 = arith.index_cast %arg3 : i32 to index
                      %64 = memref.load %alloca_0[%63] : memref<256xmemref<?xi8>>
                      %65 = "polygeist.memref2pointer"(%64) : (memref<?xi8>) -> !llvm.ptr
                      %66 = llvm.mlir.zero : !llvm.ptr
                      %67 = llvm.icmp "eq" %65, %66 : !llvm.ptr
                      %68:3 = scf.if %67 -> (i32, i32, f64) {
                        scf.yield %arg3, %arg5, %arg4 : i32, i32, f64
                      } else {
                        %69 = arith.addi %arg3, %c1_i32 : i32
                        %70 = func.call @calculate_df(%44, %arg1, %arg2) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i32
                        %71 = arith.cmpi ne, %70, %c0_i32 : i32
                        %72 = scf.if %71 -> (f64) {
                          %73 = func.call @calculate_idf(%70, %arg2) : (i32, i32) -> f64
                          %74 = func.call @lower_dup(%44) : (memref<?xi8>) -> memref<?xi8>
                          %75 = "polygeist.memref2pointer"(%74) : (memref<?xi8>) -> !llvm.ptr
                          %76 = llvm.mlir.zero : !llvm.ptr
                          %77 = llvm.icmp "eq" %75, %76 : !llvm.ptr
                          %78 = arith.xori %77, %true : i1
                          scf.if %78 {
                            func.call @score_term_over_docs(%74, %alloc_2, %alloc, %24, %73, %alloc_3, %arg2, %arg2) : (memref<?xi8>, memref<?xmemref<?xi8>>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i32, f64)>>, i32, i32) -> ()
                            memref.dealloc %74 : memref<?xi8>
                          }
                          scf.yield %73 : f64
                        } else {
                          scf.yield %arg4 : f64
                        }
                        scf.yield %69, %70, %72 : i32, i32, f64
                      }
                      scf.yield %68#0, %68#1, %68#2 : i32, i32, f64
                    } else {
                      scf.yield %arg3, %arg5, %arg4 : i32, i32, f64
                    }
                    scf.yield %52#0, %52#1, %59, %60#0, %60#1, %60#2 : i32, i32, i1, i32, i32, f64
                  } else {
                    scf.yield %arg6, %arg7, %true, %arg3, %arg5, %arg4 : i32, i32, i1, i32, i32, f64
                  }
                  scf.yield %51#0, %51#1, %51#2, %51#3, %51#4, %51#5 : i32, i32, i1, i32, i32, f64
                }
                scf.yield %47#5, %47#4, %47#0, %47#1, %47#3, %47#2 : f64, i32, i32, i32, i32, i1
              }
              memref.dealloc %33 : memref<?xi8>
              %41 = arith.index_cast %40#0 : i32 to index
              scf.for %arg3 = %c0 to %41 step %c1 {
                %43 = memref.load %alloca_0[%arg3] : memref<256xmemref<?xi8>>
                memref.dealloc %43 : memref<?xi8>
              }
              %42 = arith.index_cast %arg2 : i32 to index
              scf.for %arg3 = %c0 to %42 step %c1 {
                %43 = memref.load %alloc_2[%arg3] : memref<?xmemref<?xi8>>
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
  func.func @generate_corpus(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> memref<?xmemref<?xi8>> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c46_i8 = arith.constant 46 : i8
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c12_i64 = arith.constant 12 : i64
    %c1_i32 = arith.constant 1 : i32
    %c47 = arith.constant 47 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : i64
    %2 = memref.get_global @"generate_corpus@static@vocab" : memref<47xmemref<?xi8>>
    %alloca = memref.alloca() : memref<memref<?xmemref<?xi8>>>
    %3 = arith.cmpi sle, %arg0, %c0_i32 : i32
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %7 = arith.cmpi sle, %arg2, %c0_i32 : i32
      scf.yield %7 : i1
    }
    %5 = scf.if %4 -> (i1) {
      scf.yield %true : i1
    } else {
      %7 = arith.cmpi slt, %arg3, %arg2 : i32
      scf.yield %7 : i1
    }
    scf.if %5 {
      %7 = llvm.mlir.zero : !llvm.ptr
      %8 = "polygeist.pointer2memref"(%7) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
      affine.store %8, %alloca[] : memref<memref<?xmemref<?xi8>>>
    } else {
      %7 = memref.get_global @"generate_corpus@static@vocab@init" : memref<1xi1>
      %8 = affine.load %7[0] : memref<1xi1>
      scf.if %8 {
        affine.store %false, %7[0] : memref<1xi1>
        %24 = llvm.mlir.addressof @str1 : !llvm.ptr
        %25 = "polygeist.pointer2memref"(%24) : (!llvm.ptr) -> memref<?xi8>
        affine.store %25, %2[0] : memref<47xmemref<?xi8>>
        %26 = llvm.mlir.addressof @str2 : !llvm.ptr
        %27 = "polygeist.pointer2memref"(%26) : (!llvm.ptr) -> memref<?xi8>
        affine.store %27, %2[1] : memref<47xmemref<?xi8>>
        %28 = llvm.mlir.addressof @str3 : !llvm.ptr
        %29 = "polygeist.pointer2memref"(%28) : (!llvm.ptr) -> memref<?xi8>
        affine.store %29, %2[2] : memref<47xmemref<?xi8>>
        %30 = llvm.mlir.addressof @str4 : !llvm.ptr
        %31 = "polygeist.pointer2memref"(%30) : (!llvm.ptr) -> memref<?xi8>
        affine.store %31, %2[3] : memref<47xmemref<?xi8>>
        %32 = llvm.mlir.addressof @str5 : !llvm.ptr
        %33 = "polygeist.pointer2memref"(%32) : (!llvm.ptr) -> memref<?xi8>
        affine.store %33, %2[4] : memref<47xmemref<?xi8>>
        %34 = llvm.mlir.addressof @str6 : !llvm.ptr
        %35 = "polygeist.pointer2memref"(%34) : (!llvm.ptr) -> memref<?xi8>
        affine.store %35, %2[5] : memref<47xmemref<?xi8>>
        %36 = llvm.mlir.addressof @str7 : !llvm.ptr
        %37 = "polygeist.pointer2memref"(%36) : (!llvm.ptr) -> memref<?xi8>
        affine.store %37, %2[6] : memref<47xmemref<?xi8>>
        %38 = llvm.mlir.addressof @str8 : !llvm.ptr
        %39 = "polygeist.pointer2memref"(%38) : (!llvm.ptr) -> memref<?xi8>
        affine.store %39, %2[7] : memref<47xmemref<?xi8>>
        %40 = llvm.mlir.addressof @str9 : !llvm.ptr
        %41 = "polygeist.pointer2memref"(%40) : (!llvm.ptr) -> memref<?xi8>
        affine.store %41, %2[8] : memref<47xmemref<?xi8>>
        %42 = llvm.mlir.addressof @str10 : !llvm.ptr
        %43 = "polygeist.pointer2memref"(%42) : (!llvm.ptr) -> memref<?xi8>
        affine.store %43, %2[9] : memref<47xmemref<?xi8>>
        %44 = llvm.mlir.addressof @str11 : !llvm.ptr
        %45 = "polygeist.pointer2memref"(%44) : (!llvm.ptr) -> memref<?xi8>
        affine.store %45, %2[10] : memref<47xmemref<?xi8>>
        %46 = llvm.mlir.addressof @str12 : !llvm.ptr
        %47 = "polygeist.pointer2memref"(%46) : (!llvm.ptr) -> memref<?xi8>
        affine.store %47, %2[11] : memref<47xmemref<?xi8>>
        %48 = llvm.mlir.addressof @str13 : !llvm.ptr
        %49 = "polygeist.pointer2memref"(%48) : (!llvm.ptr) -> memref<?xi8>
        affine.store %49, %2[12] : memref<47xmemref<?xi8>>
        %50 = llvm.mlir.addressof @str14 : !llvm.ptr
        %51 = "polygeist.pointer2memref"(%50) : (!llvm.ptr) -> memref<?xi8>
        affine.store %51, %2[13] : memref<47xmemref<?xi8>>
        %52 = llvm.mlir.addressof @str15 : !llvm.ptr
        %53 = "polygeist.pointer2memref"(%52) : (!llvm.ptr) -> memref<?xi8>
        affine.store %53, %2[14] : memref<47xmemref<?xi8>>
        %54 = llvm.mlir.addressof @str16 : !llvm.ptr
        %55 = "polygeist.pointer2memref"(%54) : (!llvm.ptr) -> memref<?xi8>
        affine.store %55, %2[15] : memref<47xmemref<?xi8>>
        %56 = llvm.mlir.addressof @str17 : !llvm.ptr
        %57 = "polygeist.pointer2memref"(%56) : (!llvm.ptr) -> memref<?xi8>
        affine.store %57, %2[16] : memref<47xmemref<?xi8>>
        %58 = llvm.mlir.addressof @str18 : !llvm.ptr
        %59 = "polygeist.pointer2memref"(%58) : (!llvm.ptr) -> memref<?xi8>
        affine.store %59, %2[17] : memref<47xmemref<?xi8>>
        %60 = llvm.mlir.addressof @str19 : !llvm.ptr
        %61 = "polygeist.pointer2memref"(%60) : (!llvm.ptr) -> memref<?xi8>
        affine.store %61, %2[18] : memref<47xmemref<?xi8>>
        %62 = llvm.mlir.addressof @str20 : !llvm.ptr
        %63 = "polygeist.pointer2memref"(%62) : (!llvm.ptr) -> memref<?xi8>
        affine.store %63, %2[19] : memref<47xmemref<?xi8>>
        %64 = llvm.mlir.addressof @str21 : !llvm.ptr
        %65 = "polygeist.pointer2memref"(%64) : (!llvm.ptr) -> memref<?xi8>
        affine.store %65, %2[20] : memref<47xmemref<?xi8>>
        %66 = llvm.mlir.addressof @str22 : !llvm.ptr
        %67 = "polygeist.pointer2memref"(%66) : (!llvm.ptr) -> memref<?xi8>
        affine.store %67, %2[21] : memref<47xmemref<?xi8>>
        affine.store %57, %2[22] : memref<47xmemref<?xi8>>
        %68 = llvm.mlir.addressof @str23 : !llvm.ptr
        %69 = "polygeist.pointer2memref"(%68) : (!llvm.ptr) -> memref<?xi8>
        affine.store %69, %2[23] : memref<47xmemref<?xi8>>
        %70 = llvm.mlir.addressof @str24 : !llvm.ptr
        %71 = "polygeist.pointer2memref"(%70) : (!llvm.ptr) -> memref<?xi8>
        affine.store %71, %2[24] : memref<47xmemref<?xi8>>
        %72 = llvm.mlir.addressof @str25 : !llvm.ptr
        %73 = "polygeist.pointer2memref"(%72) : (!llvm.ptr) -> memref<?xi8>
        affine.store %73, %2[25] : memref<47xmemref<?xi8>>
        %74 = llvm.mlir.addressof @str26 : !llvm.ptr
        %75 = "polygeist.pointer2memref"(%74) : (!llvm.ptr) -> memref<?xi8>
        affine.store %75, %2[26] : memref<47xmemref<?xi8>>
        %76 = llvm.mlir.addressof @str27 : !llvm.ptr
        %77 = "polygeist.pointer2memref"(%76) : (!llvm.ptr) -> memref<?xi8>
        affine.store %77, %2[27] : memref<47xmemref<?xi8>>
        %78 = llvm.mlir.addressof @str28 : !llvm.ptr
        %79 = "polygeist.pointer2memref"(%78) : (!llvm.ptr) -> memref<?xi8>
        affine.store %79, %2[28] : memref<47xmemref<?xi8>>
        %80 = llvm.mlir.addressof @str29 : !llvm.ptr
        %81 = "polygeist.pointer2memref"(%80) : (!llvm.ptr) -> memref<?xi8>
        affine.store %81, %2[29] : memref<47xmemref<?xi8>>
        %82 = llvm.mlir.addressof @str30 : !llvm.ptr
        %83 = "polygeist.pointer2memref"(%82) : (!llvm.ptr) -> memref<?xi8>
        affine.store %83, %2[30] : memref<47xmemref<?xi8>>
        %84 = llvm.mlir.addressof @str31 : !llvm.ptr
        %85 = "polygeist.pointer2memref"(%84) : (!llvm.ptr) -> memref<?xi8>
        affine.store %85, %2[31] : memref<47xmemref<?xi8>>
        %86 = llvm.mlir.addressof @str32 : !llvm.ptr
        %87 = "polygeist.pointer2memref"(%86) : (!llvm.ptr) -> memref<?xi8>
        affine.store %87, %2[32] : memref<47xmemref<?xi8>>
        %88 = llvm.mlir.addressof @str33 : !llvm.ptr
        %89 = "polygeist.pointer2memref"(%88) : (!llvm.ptr) -> memref<?xi8>
        affine.store %89, %2[33] : memref<47xmemref<?xi8>>
        %90 = llvm.mlir.addressof @str34 : !llvm.ptr
        %91 = "polygeist.pointer2memref"(%90) : (!llvm.ptr) -> memref<?xi8>
        affine.store %91, %2[34] : memref<47xmemref<?xi8>>
        %92 = llvm.mlir.addressof @str35 : !llvm.ptr
        %93 = "polygeist.pointer2memref"(%92) : (!llvm.ptr) -> memref<?xi8>
        affine.store %93, %2[35] : memref<47xmemref<?xi8>>
        %94 = llvm.mlir.addressof @str36 : !llvm.ptr
        %95 = "polygeist.pointer2memref"(%94) : (!llvm.ptr) -> memref<?xi8>
        affine.store %95, %2[36] : memref<47xmemref<?xi8>>
        %96 = llvm.mlir.addressof @str37 : !llvm.ptr
        %97 = "polygeist.pointer2memref"(%96) : (!llvm.ptr) -> memref<?xi8>
        affine.store %97, %2[37] : memref<47xmemref<?xi8>>
        %98 = llvm.mlir.addressof @str38 : !llvm.ptr
        %99 = "polygeist.pointer2memref"(%98) : (!llvm.ptr) -> memref<?xi8>
        affine.store %99, %2[38] : memref<47xmemref<?xi8>>
        %100 = llvm.mlir.addressof @str39 : !llvm.ptr
        %101 = "polygeist.pointer2memref"(%100) : (!llvm.ptr) -> memref<?xi8>
        affine.store %101, %2[39] : memref<47xmemref<?xi8>>
        %102 = llvm.mlir.addressof @str40 : !llvm.ptr
        %103 = "polygeist.pointer2memref"(%102) : (!llvm.ptr) -> memref<?xi8>
        affine.store %103, %2[40] : memref<47xmemref<?xi8>>
        %104 = llvm.mlir.addressof @str41 : !llvm.ptr
        %105 = "polygeist.pointer2memref"(%104) : (!llvm.ptr) -> memref<?xi8>
        affine.store %105, %2[41] : memref<47xmemref<?xi8>>
        %106 = llvm.mlir.addressof @str42 : !llvm.ptr
        %107 = "polygeist.pointer2memref"(%106) : (!llvm.ptr) -> memref<?xi8>
        affine.store %107, %2[42] : memref<47xmemref<?xi8>>
        %108 = llvm.mlir.addressof @str43 : !llvm.ptr
        %109 = "polygeist.pointer2memref"(%108) : (!llvm.ptr) -> memref<?xi8>
        affine.store %109, %2[43] : memref<47xmemref<?xi8>>
        %110 = llvm.mlir.addressof @str44 : !llvm.ptr
        %111 = "polygeist.pointer2memref"(%110) : (!llvm.ptr) -> memref<?xi8>
        affine.store %111, %2[44] : memref<47xmemref<?xi8>>
        %112 = llvm.mlir.addressof @str45 : !llvm.ptr
        %113 = "polygeist.pointer2memref"(%112) : (!llvm.ptr) -> memref<?xi8>
        affine.store %113, %2[45] : memref<47xmemref<?xi8>>
        %114 = llvm.mlir.addressof @str46 : !llvm.ptr
        %115 = "polygeist.pointer2memref"(%114) : (!llvm.ptr) -> memref<?xi8>
        affine.store %115, %2[46] : memref<47xmemref<?xi8>>
      }
      %9 = "polygeist.typeSize"() <{source = memref<?xi8>}> : () -> index
      %10 = arith.muli %9, %c47 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = arith.index_cast %9 : index to i64
      %13 = arith.divui %11, %12 : i64
      %14 = arith.trunci %13 : i64 to i32
      func.call @srand(%arg1) : (i32) -> ()
      %15 = "polygeist.typeSize"() <{source = memref<?xi8>}> : () -> index
      %16 = arith.extsi %arg0 : i32 to i64
      %17 = arith.index_cast %15 : index to i64
      %18 = arith.muli %16, %17 : i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.divui %19, %15 : index
      %alloc = memref.alloc(%20) : memref<?xmemref<?xi8>>
      %21 = "polygeist.memref2pointer"(%alloc) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %22 = llvm.mlir.zero : !llvm.ptr
      %23 = llvm.icmp "eq" %21, %22 : !llvm.ptr
      scf.if %23 {
        %24 = "polygeist.pointer2memref"(%22) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
        affine.store %24, %alloca[] : memref<memref<?xmemref<?xi8>>>
      } else {
        %24:6 = scf.while (%arg4 = %0, %arg5 = %0, %arg6 = %1, %arg7 = %0, %arg8 = %c0_i32, %arg9 = %true, %arg10 = %true) : (i32, i32, i64, i32, i32, i1, i1) -> (i1, i32, i32, i64, i32, i32) {
          %25 = arith.cmpi slt, %arg8, %arg0 : i32
          %26 = arith.andi %25, %arg10 : i1
          scf.condition(%26) %arg9, %arg4, %arg5, %arg6, %arg7, %arg8 : i1, i32, i32, i64, i32, i32
        } do {
        ^bb0(%arg4: i1, %arg5: i32, %arg6: i32, %arg7: i64, %arg8: i32, %arg9: i32):
          %25 = func.call @rand() : () -> i32
          %26 = arith.subi %arg3, %arg2 : i32
          %27 = arith.addi %26, %c1_i32 : i32
          %28 = arith.remsi %25, %27 : i32
          %29 = arith.addi %arg2, %28 : i32
          %30 = arith.extsi %29 : i32 to i64
          %31 = arith.muli %30, %c12_i64 : i64
          %32 = arith.addi %31, %c2_i64 : i64
          %33 = arith.index_cast %32 : i64 to index
          %alloc_0 = memref.alloc(%33) : memref<?xi8>
          %34 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xi8>) -> !llvm.ptr
          %35 = llvm.mlir.zero : !llvm.ptr
          %36 = llvm.icmp "eq" %34, %35 : !llvm.ptr
          %37 = arith.xori %36, %true : i1
          %38 = arith.andi %37, %arg4 : i1
          %39 = arith.xori %36, %true : i1
          %40 = arith.select %36, %arg9, %arg8 : i32
          %41:4 = scf.if %36 -> (i32, i32, i64, i32) {
            %42 = arith.index_cast %arg9 : i32 to index
            scf.for %arg10 = %c0 to %42 step %c1 {
              %44 = memref.load %alloc[%arg10] : memref<?xmemref<?xi8>>
              memref.dealloc %44 : memref<?xi8>
            }
            memref.dealloc %alloc : memref<?xmemref<?xi8>>
            %43 = "polygeist.pointer2memref"(%35) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
            affine.store %43, %alloca[] : memref<memref<?xmemref<?xi8>>>
            scf.yield %arg5, %arg6, %arg7, %arg9 : i32, i32, i64, i32
          } else {
            %42:3 = scf.while (%arg10 = %arg5, %arg11 = %c0_i32, %arg12 = %c0_i64, %arg13 = %true) : (i32, i32, i64, i1) -> (i32, i32, i64) {
              %48 = arith.cmpi slt, %arg11, %29 : i32
              %49 = arith.andi %48, %arg13 : i1
              scf.condition(%49) %arg10, %arg11, %arg12 : i32, i32, i64
            } do {
            ^bb0(%arg10: i32, %arg11: i32, %arg12: i64):
              %48 = func.call @rand() : () -> i32
              %49 = arith.remsi %48, %14 : i32
              %50 = arith.index_cast %49 : i32 to index
              %51 = memref.load %2[%50] : memref<47xmemref<?xi8>>
              %52 = arith.cmpi eq, %arg11, %c0_i32 : i32
              %53 = scf.if %52 -> (i32) {
                %62 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xi8>) -> !llvm.ptr
                %63 = llvm.getelementptr %62[%arg12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                %64 = arith.subi %32, %arg12 : i64
                %65 = llvm.mlir.addressof @str47 : !llvm.ptr
                %66 = llvm.getelementptr %65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
                %67 = "polygeist.memref2pointer"(%51) : (memref<?xi8>) -> !llvm.ptr
                %68 = llvm.call @snprintf(%63, %64, %66, %67) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
                scf.yield %68 : i32
              } else {
                %62 = "polygeist.memref2pointer"(%alloc_0) : (memref<?xi8>) -> !llvm.ptr
                %63 = llvm.getelementptr %62[%arg12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                %64 = arith.subi %32, %arg12 : i64
                %65 = llvm.mlir.addressof @str48 : !llvm.ptr
                %66 = llvm.getelementptr %65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
                %67 = "polygeist.memref2pointer"(%51) : (memref<?xi8>) -> !llvm.ptr
                %68 = llvm.call @snprintf(%63, %64, %66, %67) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
                scf.yield %68 : i32
              }
              %54 = arith.cmpi slt, %53, %c0_i32 : i32
              %55 = arith.select %54, %c0_i32, %53 : i32
              %56 = arith.extsi %55 : i32 to i64
              %57 = arith.addi %arg12, %56 : i64
              %58 = arith.cmpi sge, %57, %32 : i64
              %59 = arith.cmpi slt, %57, %32 : i64
              %60 = scf.if %58 -> (i64) {
                %62 = arith.addi %31, %c1_i64 : i64
                scf.yield %62 : i64
              } else {
                scf.yield %57 : i64
              }
              %61 = scf.if %59 -> (i32) {
                %62 = arith.addi %arg11, %c1_i32 : i32
                scf.yield %62 : i32
              } else {
                scf.yield %arg11 : i32
              }
              scf.yield %55, %61, %60, %59 : i32, i32, i64, i1
            }
            %43 = arith.addi %42#2, %c1_i64 : i64
            %44 = arith.cmpi slt, %43, %32 : i64
            %45 = arith.select %44, %43, %42#2 : i64
            scf.if %44 {
              %48 = arith.index_cast %42#2 : i64 to index
              memref.store %c46_i8, %alloc_0[%48] : memref<?xi8>
              %49 = arith.index_cast %43 : i64 to index
              memref.store %c0_i8, %alloc_0[%49] : memref<?xi8>
            } else {
              %48 = arith.addi %31, %c1_i64 : i64
              %49 = arith.index_cast %48 : i64 to index
              memref.store %c0_i8, %alloc_0[%49] : memref<?xi8>
            }
            %46 = arith.index_cast %arg9 : i32 to index
            memref.store %alloc_0, %alloc[%46] : memref<?xmemref<?xi8>>
            %47 = arith.addi %arg9, %c1_i32 : i32
            scf.yield %42#0, %42#1, %45, %47 : i32, i32, i64, i32
          }
          scf.yield %41#0, %41#1, %41#2, %40, %41#3, %38, %39 : i32, i32, i64, i32, i32, i1, i1
        }
        scf.if %24#0 {
          affine.store %alloc, %alloca[] : memref<memref<?xmemref<?xi8>>>
        }
      }
    }
    %6 = affine.load %alloca[] : memref<memref<?xmemref<?xi8>>>
    return %6 : memref<?xmemref<?xi8>>
  }
  func.func private @srand(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @free_corpus(%arg0: memref<?xmemref<?xi8>>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.icmp "eq" %0, %1 : !llvm.ptr
    %3 = arith.xori %2, %true : i1
    scf.if %3 {
      %4 = arith.index_cast %arg1 : i32 to index
      scf.for %arg2 = %c0 to %4 step %c1 {
        %5 = memref.load %arg0[%arg2] : memref<?xmemref<?xi8>>
        memref.dealloc %5 : memref<?xi8>
      }
      memref.dealloc %arg0 : memref<?xmemref<?xi8>>
    }
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %cst = arith.constant 1.000000e+06 : f64
    %cst_0 = arith.constant 1.000000e+03 : f64
    %c16_i32 = arith.constant 16 : i32
    %c4_i32 = arith.constant 4 : i32
    %c10_i32 = arith.constant 10 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c42_i32 = arith.constant 42 : i32
    %c6_i32 = arith.constant 6 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1x2xi64>
    %alloca_1 = memref.alloca() : memref<1x2xi64>
    %alloca_2 = memref.alloca() : memref<memref<?xmemref<?xi8>>>
    %alloca_3 = memref.alloca() : memref<memref<?xi8>>
    %1 = arith.cmpi sge, %arg0, %c2_i32 : i32
    %2:3 = scf.if %1 -> (i32, i1, i32) {
      %5 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %6 = func.call @atoi(%5) : (memref<?xi8>) -> i32
      %7 = arith.cmpi sle, %6, %c0_i32 : i32
      %8 = arith.cmpi sgt, %6, %c0_i32 : i32
      %9 = arith.select %7, %c1_i32, %0 : i32
      scf.if %7 {
        %10 = llvm.mlir.addressof @stderr : !llvm.ptr
        %11 = llvm.load %10 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
        %12 = "polygeist.memref2pointer"(%11) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
        %13 = llvm.mlir.addressof @str49 : !llvm.ptr
        %14 = llvm.getelementptr %13[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
        %15 = llvm.call @fprintf(%12, %14) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        %16 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
        func.call @usage(%16) : (memref<?xi8>) -> ()
      }
      scf.yield %6, %8, %9 : i32, i1, i32
    } else {
      scf.yield %c6_i32, %true, %0 : i32, i1, i32
    }
    %3:2 = scf.if %2#1 -> (i1, i32) {
      %5 = arith.cmpi sge, %arg0, %c3_i32 : i32
      %6 = scf.if %5 -> (i32) {
        %22 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %23 = llvm.mlir.zero : !llvm.ptr
        %24 = "polygeist.pointer2memref"(%23) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
        %25 = func.call @strtoul(%22, %24, %c10_i32) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i64
        %26 = arith.trunci %25 : i64 to i32
        scf.yield %26 : i32
      } else {
        scf.yield %c42_i32 : i32
      }
      %7 = arith.cmpi sge, %arg0, %c4_i32 : i32
      scf.if %7 {
        %22 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
        affine.store %22, %alloca_3[] : memref<memref<?xi8>>
      }
      %8 = llvm.mlir.addressof @str50 : !llvm.ptr
      %9 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
      %10 = affine.load %alloca_3[] : memref<memref<?xi8>>
      %11 = "polygeist.memref2pointer"(%10) : (memref<?xi8>) -> !llvm.ptr
      %12 = llvm.call @printf(%9, %11) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %13 = llvm.mlir.addressof @str51 : !llvm.ptr
      %14 = llvm.getelementptr %13[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %15 = llvm.call @printf(%14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %16 = func.call @generate_corpus(%2#0, %6, %c6_i32, %c16_i32) : (i32, i32, i32, i32) -> memref<?xmemref<?xi8>>
      affine.store %16, %alloca_2[] : memref<memref<?xmemref<?xi8>>>
      %17 = "polygeist.memref2pointer"(%16) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %18 = llvm.mlir.zero : !llvm.ptr
      %19 = llvm.icmp "eq" %17, %18 : !llvm.ptr
      %20 = arith.xori %19, %true : i1
      %21 = arith.select %19, %c1_i32, %2#2 : i32
      scf.if %19 {
        %22 = llvm.mlir.addressof @stderr : !llvm.ptr
        %23 = llvm.load %22 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
        %24 = "polygeist.memref2pointer"(%23) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
        %25 = llvm.mlir.addressof @str52 : !llvm.ptr
        %26 = llvm.getelementptr %25[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
        %27 = llvm.call @fprintf(%24, %26) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      }
      scf.yield %20, %21 : i1, i32
    } else {
      scf.yield %false, %2#2 : i1, i32
    }
    %4 = arith.select %3#0, %c0_i32, %3#1 : i32
    scf.if %3#0 {
      %cast = memref.cast %alloca_1 : memref<1x2xi64> to memref<?x2xi64>
      %5 = func.call @clock_gettime(%c1_i32, %cast) : (i32, memref<?x2xi64>) -> i32
      %6 = affine.load %alloca_3[] : memref<memref<?xi8>>
      %7 = affine.load %alloca_2[] : memref<memref<?xmemref<?xi8>>>
      %8 = func.call @rank_documents_bm25(%6, %7, %2#0) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> memref<?x!llvm.struct<(i32, f64)>>
      %cast_4 = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
      %9 = func.call @clock_gettime(%c1_i32, %cast_4) : (i32, memref<?x2xi64>) -> i32
      %10 = affine.load %alloca[0, 0] : memref<1x2xi64>
      %11 = affine.load %alloca_1[0, 0] : memref<1x2xi64>
      %12 = arith.subi %10, %11 : i64
      %13 = arith.sitofp %12 : i64 to f64
      %14 = arith.mulf %13, %cst_0 : f64
      %15 = affine.load %alloca[0, 1] : memref<1x2xi64>
      %16 = affine.load %alloca_1[0, 1] : memref<1x2xi64>
      %17 = arith.subi %15, %16 : i64
      %18 = arith.sitofp %17 : i64 to f64
      %19 = arith.divf %18, %cst : f64
      %20 = arith.addf %14, %19 : f64
      %21 = "polygeist.memref2pointer"(%8) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
      %22 = llvm.mlir.zero : !llvm.ptr
      %23 = llvm.icmp "ne" %21, %22 : !llvm.ptr
      scf.if %23 {
        %27 = arith.index_cast %2#0 : i32 to index
        scf.for %arg2 = %c0 to %27 step %c1 {
          %28 = arith.index_cast %arg2 : index to i32
          %29 = arith.muli %arg2, %c16 : index
          %30 = arith.index_cast %29 : index to i64
          %31 = llvm.getelementptr %21[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %32 = llvm.load %31 : !llvm.ptr -> i32
          %33 = llvm.mlir.addressof @str53 : !llvm.ptr
          %34 = llvm.getelementptr %33[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
          %35 = arith.addi %28, %c1_i32 : i32
          %36 = llvm.getelementptr %31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
          %37 = llvm.load %36 : !llvm.ptr -> f64
          %38 = arith.index_cast %32 : i32 to index
          %39 = memref.load %7[%38] : memref<?xmemref<?xi8>>
          %40 = "polygeist.memref2pointer"(%39) : (memref<?xi8>) -> !llvm.ptr
          %41 = llvm.call @printf(%34, %35, %32, %37, %40) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, f64, !llvm.ptr) -> i32
        }
        memref.dealloc %8 : memref<?x!llvm.struct<(i32, f64)>>
      } else {
        %27 = llvm.mlir.addressof @str54 : !llvm.ptr
        %28 = llvm.getelementptr %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
        %29 = llvm.call @printf(%28) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      }
      %24 = llvm.mlir.addressof @str55 : !llvm.ptr
      %25 = llvm.getelementptr %24[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
      %26 = llvm.call @printf(%25, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    }
    return %4 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @usage(%arg0: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %2 = "polygeist.memref2pointer"(%1) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %3 = llvm.mlir.addressof @str56 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<46 x i8>
    %5 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %6 = llvm.call @fprintf(%2, %4, %5) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %7 = llvm.load %0 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %8 = "polygeist.memref2pointer"(%7) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %9 = llvm.mlir.addressof @str57 : !llvm.ptr
    %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<62 x i8>
    %11 = llvm.call @fprintf(%8, %10) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %12 = llvm.load %0 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %13 = "polygeist.memref2pointer"(%12) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %14 = llvm.mlir.addressof @str58 : !llvm.ptr
    %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<40 x i8>
    %16 = llvm.call @fprintf(%13, %15) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %17 = llvm.load %0 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %18 = "polygeist.memref2pointer"(%17) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %19 = llvm.mlir.addressof @str59 : !llvm.ptr
    %20 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<57 x i8>
    %21 = llvm.call @fprintf(%18, %20) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    return
  }
  func.func private @strtoul(memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @clock_gettime(i32, memref<?x2xi64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}

