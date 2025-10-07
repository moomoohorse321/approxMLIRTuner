BENCH=pagerank # e.g. bm25
ROOT=${2:-/home/yimu3/PolygeistSample} # e.g. /home/hao/Polygeist
/home/yimu3/PolygeistSample/build/bin/cgeist -O0 -S -resource-dir=/home/yimu3/PolygeistSample/llvm-project/build/clang/18/ -I /home/yimu3/PolygeistSample/tools/cgeist/Test/polybench/utilities /home/yimu3/PolygeistSample/tools/cgeist/Test/approxMLIR/approx_pagerank.c > /home/yimu3/PolygeistSample/tools/cgeist/Test/approxMLIR/approx_pagerank.c.mlir
bash /home/yimu3/PolygeistSample/tools/cgeist/Test/approxMLIR/merge.bash pagerank # depend on existence of /home/yimu3/PolygeistSample/build/tools/cgeist/Test/approxMLIR/approx_pagerank.c.mlir, /home/yimu3/PolygeistSample/build/tools/cgeist/Test/approxMLIR/pagerank/annotation.mlir

# auto-tuner config (use your regex to configure all the annotations)
# /home/yimu3/PolygeistSample/build/bin/polygeist-opt /home/yimu3/PolygeistSample//tools/cgeist/Test/approxMLIR/approx_pagerank.mlir -pre-emit-transform -emit-approx -config-approx -transform-approx > test.mlir
# /home/yimu3/PolygeistSample/build/bin/cgeist -resource-dir=/home/yimu3/PolygeistSample/llvm-project/build/lib/clang/18 -I /home/yimu3/PolygeistSample/tools/cgeist/Test/polybench/utilities -lm test.mlir -import-mlir -o test.exec