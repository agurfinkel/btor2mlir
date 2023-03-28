; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @__VERIFIER_error()

define void @main() !dbg !3 {
  br label %1, !dbg !7

1:                                                ; preds = %8, %0
  %2 = phi <16 x i4> [ %5, %8 ], [ <i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1>, %0 ]
  %3 = extractelement <16 x i4> %2, i4 -8, !dbg !9
  %4 = add i4 %3, 1, !dbg !10
  %5 = insertelement <16 x i4> %2, i4 %4, i4 -8, !dbg !11
  %6 = icmp eq i4 %3, -1, !dbg !12
  %7 = xor i1 %6, true, !dbg !13
  br i1 %7, label %8, label %9, !dbg !14

8:                                                ; preds = %1
  br label %1, !dbg !15

9:                                                ; preds = %1
  call void @__VERIFIER_error(), !dbg !16
  unreachable, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 3, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "btor2mlir-1/test/Btor/array.btor2.mlir.opt", directory: "/home/jetafese")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 5, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 9, column: 10, scope: !8)
!10 = !DILocation(line: 10, column: 10, scope: !8)
!11 = !DILocation(line: 11, column: 10, scope: !8)
!12 = !DILocation(line: 13, column: 10, scope: !8)
!13 = !DILocation(line: 15, column: 11, scope: !8)
!14 = !DILocation(line: 16, column: 5, scope: !8)
!15 = !DILocation(line: 18, column: 5, scope: !8)
!16 = !DILocation(line: 20, column: 5, scope: !8)
!17 = !DILocation(line: 21, column: 5, scope: !8)
