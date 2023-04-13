; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @__VERIFIER_error()

declare void @btor2mlir_print_input_num(i64, i64, i64)

declare i8 @nd_bv8()

define void @main() !dbg !3 {
  br label %1, !dbg !7

1:                                                ; preds = %6, %0
  %2 = phi i1 [ %4, %6 ], [ false, %0 ]
  %3 = call i8 @nd_bv8(), !dbg !9
  %4 = icmp eq i8 %3, -1, !dbg !10
  %5 = xor i1 %2, true, !dbg !11
  br i1 %5, label %6, label %7, !dbg !12

6:                                                ; preds = %1
  br label %1, !dbg !13

7:                                                ; preds = %1
  call void @__VERIFIER_error(), !dbg !14
  unreachable, !dbg !15
}

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
!10 = !DILocation(line: 12, column: 10, scope: !8)
!11 = !DILocation(line: 14, column: 10, scope: !8)
!12 = !DILocation(line: 15, column: 5, scope: !8)
!13 = !DILocation(line: 17, column: 5, scope: !8)
!14 = !DILocation(line: 19, column: 5, scope: !8)
!15 = !DILocation(line: 20, column: 5, scope: !8)
