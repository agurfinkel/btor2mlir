// RUN: btor2mlir-opt %s | btor2mlir-opt | FileCheck %s

module {
    func @init( ) -> (i3,i3) {
        %0 = btor.const 0 : i3
        %1 = btor.const 0 : i3
        return %0, %1 : i3, i3
    }

    func @input() -> (i3, i3) {
        %const = btor.const 7 : i3
        %0 = btor.input 0, %const :  i3
        %1 = btor.input 9, %const : i3
        return %0, %1 : i3, i3
    }

    func @next( %arg0: i3, %arg1: i3 ) -> (i3, i3) {
        // create assumption
        %cmp_ne = btor.cmp "ne", %arg0, %arg1 : i3
        btor.assume ( %cmp_ne )
        // apply transition relation
        %c_0 = btor.const 1 : i3
        %add_1 = btor.add %arg0, %c_0 : i3
        %sub_1 = btor.sub %arg1, %c_0 : i3
        // create assertion
        %bad = btor.cmp "ne", %add_1, %sub_1 : i3
        %notbad = btor.not %bad : i1
        btor.assert ( %notbad )
        return %add_1, %sub_1 : i3, i3
    }
}
