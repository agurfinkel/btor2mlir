module attributes {llvm.data_layout = ""}  {
  llvm.func @memrefCopy(i64, !llvm.ptr<struct<(i64, ptr<i8>)>>, !llvm.ptr<struct<(i64, ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(8 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.null : !llvm.ptr<i32>
    %3 = llvm.getelementptr %2[%0] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %4 = llvm.ptrtoint %3 : !llvm.ptr<i32> to i64
    %5 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr<i8>
    %6 = llvm.bitcast %5 : !llvm.ptr<i8> to !llvm.ptr<i32>
    %7 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %6, %8[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.insertvalue %10, %9[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %0, %11[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %1, %12[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = builtin.unrealized_conversion_cast %13 : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> to memref<8xi32>
    %15 = builtin.unrealized_conversion_cast %14 : memref<8xi32> to !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%15 : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb1(%16: !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb0, ^bb1
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> to memref<8xi32>
    %18 = llvm.mlir.constant(1 : i8) : i8
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = builtin.unrealized_conversion_cast %19 : i64 to index
    %21 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.getelementptr %21[%19] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %23 = llvm.load %22 : !llvm.ptr<i32>
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %26 = llvm.mlir.constant(8 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.null : !llvm.ptr<i32>
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<i32> to i64
    %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr<i8>
    %32 = llvm.bitcast %31 : !llvm.ptr<i8> to !llvm.ptr<i32>
    %33 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.insertvalue %32, %33[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %32, %34[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.mlir.constant(0 : index) : i64
    %37 = llvm.insertvalue %36, %35[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.insertvalue %26, %37[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %27, %38[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %40 = builtin.unrealized_conversion_cast %39 : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> to memref<8xi32>
    %41 = builtin.unrealized_conversion_cast %40 : memref<8xi32> to !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.mlir.constant(1 : index) : i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.alloca %43 x !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %16, %44 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    %45 = llvm.bitcast %44 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %46 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %47 = llvm.insertvalue %42, %46[0] : !llvm.struct<(i64, ptr<i8>)>
    %48 = llvm.insertvalue %45, %47[1] : !llvm.struct<(i64, ptr<i8>)>
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.alloca %50 x !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %39, %51 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    %52 = llvm.bitcast %51 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %53 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %54 = llvm.insertvalue %49, %53[0] : !llvm.struct<(i64, ptr<i8>)>
    %55 = llvm.insertvalue %52, %54[1] : !llvm.struct<(i64, ptr<i8>)>
    %56 = llvm.mlir.constant(1 : index) : i64
    %57 = llvm.alloca %56 x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.store %48, %57 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %58 = llvm.alloca %56 x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.store %55, %58 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %59 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%59, %57, %58) : (i64, !llvm.ptr<struct<(i64, ptr<i8>)>>, !llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
    %60 = llvm.extractvalue %39[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %61 = llvm.getelementptr %60[%24] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %23, %61 : !llvm.ptr<i32>
    llvm.br ^bb1(%41 : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)
  }
}

