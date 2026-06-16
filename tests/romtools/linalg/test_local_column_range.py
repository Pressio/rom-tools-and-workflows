import numpy as np

from romtools.linalg.linalg import _local_column_range

def test_python_local_column_range_r0_s4_m8():
    rank = 0
    size = 4
    M = 8

    start, end = _local_column_range(rank=rank, size=size, M=M)

    gold_start = 0
    gold_end = 2

    assert gold_start == start
    assert gold_end == end

def test_python_local_column_range_r1_s1_m8():
    rank = 1
    size = 1
    M = 8

    start, end = _local_column_range(rank=rank, size=size, M=M)

    gold_start = 8
    gold_end = 16

    assert gold_start == start
    assert gold_end == end

def test_python_local_column_range_r1_s2_m8():
    rank = 1
    size = 2
    M = 8

    start, end = _local_column_range(rank=rank, size=size, M=M)

    gold_start = 4
    gold_end = 8

    assert gold_start == start
    assert gold_end == end

def test_python_local_column_range_r2_s1_m8():
    rank = 2
    size = 1
    M = 8

    start, end = _local_column_range(rank=rank, size=size, M=M)

    gold_start = 16
    gold_end = 24

    assert gold_start == start
    assert gold_end == end

def test_python_local_column_range_r2_s3_m8():
    rank = 2
    size = 3
    M = 8

    start, end = _local_column_range(rank=rank, size=size, M=M)

    gold_start = 6
    gold_end = 8

    assert gold_start == start
    assert gold_end == end

def test_python_local_column_range_r3_s2_m8():
    rank = 3
    size = 2
    M = 8

    start, end = _local_column_range(rank=rank, size=size, M=M)

    gold_start = 12
    gold_end = 16

    assert gold_start == start
    assert gold_end == end

if __name__ == "__main__":
    test_python_local_column_range_r0_s4_m8()

    test_python_local_column_range_r1_s1_m8()
    test_python_local_column_range_r1_s2_m8()

    test_python_local_column_range_r2_s1_m8()
    test_python_local_column_range_r2_s3_m8()

    test_python_local_column_range_r3_s2_m8()
