// RUN: btor2mlir-opt %s | btor2mlir-opt | FileCheck %s

module {
    func.func @next( %arg0: i3, %arg1: i3 ) -> (i3, i3) {
        // create assumption
        %cmp_ne = btor.cmp "ne", %arg0, %arg1 : i3
        btor.constraint ( %cmp_ne )
        // apply transition relation
        %c_0 = btor.constant 1 : i3
        %add_1 = btor.add %arg0, %c_0 : i3
        %sub_1 = btor.sub %arg1, %c_0 : i3
        // create assertion
        %bad = btor.cmp "ne", %add_1, %sub_1 : i3
        %redop = btor.redand %bad : i1
        btor.assert_not ( %bad )
        return %add_1, %sub_1 : i3, i3
    }
}
