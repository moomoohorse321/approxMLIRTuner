BENCH=blacksholes # e.g. bm25
ROOT=${2:-/home/yimu3/PolygeistSample} # e.g. /home/hao/Polygeist
${ROOT}/build/bin/cgeist -O0 -S -resource-dir=${ROOT}/llvm-project/build/lib/clang/18 -I ${ROOT}/tools/cgeist/Test/polybench/utilities ${ROOT}/tools/cgeist/Test/approxMLIR/approx_${BENCH}.c > ${ROOT}/tools/cgeist/Test/approxMLIR/approx_${BENCH}.c.mlir
bash ${ROOT}/tools/cgeist/Test/approxMLIR/merge.bash ${BENCH} # depend on existence of ${ROOT}/build/tools/cgeist/Test/approxMLIR/approx_${BENCH}.c.mlir, ${ROOT}/build/tools/cgeist/Test/approxMLIR/${BENCH}/annotation.mlir

# auto-tuner config (use your regex to configure all the annotations)
# ${ROOT}/build/bin/polygeist-opt ${ROOT}//tools/cgeist/Test/approxMLIR/approx_${BENCH}.mlir -pre-emit-transform -emit-approx -config-approx -transform-approx > test.mlir
# ${ROOT}/build/bin/cgeist -resource-dir=${ROOT}/llvm-project/build/lib/clang/18 -I ${ROOT}/tools/cgeist/Test/polybench/utilities -lm test.mlir -import-mlir -o test.exec