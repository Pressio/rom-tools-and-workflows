
from romtools.vector_space import *
from romtools.composite_vector_space import *

from romtools.vector_space.utils import *
import numpy as np 


def test_basic_composite_vector_space():
    ## Basic check 
    #=======================================
    snapshots = np.random.normal(size=(5,8,4))
    v1 = VectorSpaceFromPOD(snapshots[0:2])
    v2 = VectorSpaceFromPOD(snapshots[2:4])
    v3 = VectorSpaceFromPOD(snapshots[4::])

    V = CompositeVectorSpace([v1,v2,v3])
    global_basis = V.get_basis()
    global_shift = V.get_shift_vector()

    v1b = v1.get_basis()
    v2b = v2.get_basis()
    v3b = v3.get_basis()
    v1v = v1.get_shift_vector()
    v2v = v2.get_shift_vector()
    v3v = v3.get_shift_vector()
    

    global_basis_2 = np.zeros((5,8,12))
    global_basis_2[0:2,:,0:4] = v1b
    global_basis_2[2:4,:,4:8] = v2b
    global_basis_2[4::,:,8::] = v3b

    global_shift_2 = np.append(v1v,v2v,axis=0)
    global_shift_2 = np.append(global_shift_2,v3v,axis=0)

    assert np.allclose(global_basis,global_basis_2)
    assert np.allclose(global_shift,global_shift_2)
    assert np.allclose(global_basis.shape, V.extents())

def test_non_uniform_composite_vector_space():
    ## Check with non-uniform sized spaces 
    #=======================================
    snapshots = np.random.normal(size=(5,8,4))
    my_truncater = BasisSizeTruncater(3) 
    v1 = VectorSpaceFromPOD(snapshots[0:2],my_truncater)
    my_truncater = BasisSizeTruncater(2) 
    v2 = VectorSpaceFromPOD(snapshots[2:4],my_truncater)
    my_truncater = BasisSizeTruncater(4) 
    v3 = VectorSpaceFromPOD(snapshots[4::],my_truncater)

    V = CompositeVectorSpace([v1,v2,v3])
    global_basis = V.get_basis()
    global_shift = V.get_shift_vector()

    v1b = v1.get_basis()
    v2b = v2.get_basis()
    v3b = v3.get_basis()
    v1v = v1.get_shift_vector()
    v2v = v2.get_shift_vector()
    v3v = v3.get_shift_vector()

    global_basis_2 = np.zeros((5,8,9))
    global_basis_2[0:2,:,0:3] = v1b
    global_basis_2[2:4,:,3:5] = v2b
    global_basis_2[4::,:,5::] = v3b

    global_shift_2 = np.append(v1v,v2v,axis=0)
    global_shift_2 = np.append(global_shift_2,v3v,axis=0)

    assert np.allclose(global_basis,global_basis_2)
    assert np.allclose(global_shift,global_shift_2)


def test_compact_composite_vector_space():
    # Test compact representation
    #=============================================
    snapshots = np.random.normal(size=(5,8,4))
    my_truncater = BasisSizeTruncater(3) 
    v1 = VectorSpaceFromPOD(snapshots[0:2],my_truncater)
    my_truncater = BasisSizeTruncater(2) 
    v2 = VectorSpaceFromPOD(snapshots[2:4],my_truncater)
    my_truncater = BasisSizeTruncater(4) 
    v3 = VectorSpaceFromPOD(snapshots[4::],my_truncater)

    V = CompositeVectorSpace([v1,v2,v3])
    global_compact_basis = V.get_compact_basis()
    global_shift = V.get_shift_vector()
    global_compact_shift = V.get_compact_shift_vector()

    v1b = v1.get_basis()
    v2b = v2.get_basis()
    v3b = v3.get_basis()
    v1v = v1.get_shift_vector()
    v2v = v2.get_shift_vector()
    v3v = v3.get_shift_vector()

    global_shift_2 = np.append(v1v,v2v,axis=0)
    global_shift_2 = np.append(global_shift_2,v3v,axis=0)
    vbs = [v1b,v2b,v3b]
    shifts = [v1v,v2v,v3v]
    for i in range(0,3):
      assert np.allclose(global_compact_basis[i],vbs[i])
      assert np.allclose(global_compact_shift[i],shifts[i])
    assert(np.allclose(global_shift,global_shift_2))


if __name__=="__main__":
    test_basic_composite_vector_space()
    test_non_uniform_composite_vector_space()
    test_compact_composite_vector_space()
  
