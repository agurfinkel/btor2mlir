; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @__VERIFIER_error()

declare void @btor2mlir_print_input_num(i64, i8)

declare i8 @nd_bv8_in3()

define void @main() !dbg !3 {
  br label %1, !dbg !7

1:                                                ; preds = %7, %0
  %2 = phi i1 [ %5, %7 ], [ false, %0 ]
  %3 = call i8 @nd_bv8_in3(), !dbg !9
  call void @btor2mlir_print_input_num(i64 3, i8 %3), !dbg !10
  %4 = bitcast i8 %3 to <8 x i1>, !dbg !11
  %5 = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %4), !dbg !12
  %6 = xor i1 %2, true, !dbg !13
  br i1 %6, label %7, label %8, !dbg !14

7:                                                ; preds = %1
  br label %1, !dbg !15

8:                                                ; preds = %1
  call void @__VERIFIER_error(), !dbg !16
  unreachable, !dbg !17
}

; Function Attrs: nofree nosync nounwind readnone willreturn
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>) #0

attributes #0 = { nofree nosync nounwind readnone willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 5, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "btor2mlir-1/test/Btor/redor.btor.mlir.opt", directory: "/home/jetafese")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 7, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 9, column: 10, scope: !8)
!10 = !DILocation(line: 11, column: 5, scope: !8)
!11 = !DILocation(line: 13, column: 10, scope: !8)
!12 = !DILocation(line: 14, column: 10, scope: !8)
!13 = !DILocation(line: 16, column: 10, scope: !8)
!14 = !DILocation(line: 17, column: 5, scope: !8)
!15 = !DILocation(line: 19, column: 5, scope: !8)
!16 = !DILocation(line: 21, column: 5, scope: !8)
!17 = !DILocation(line: 22, column: 5, scope: !8)
