'''
Helper functions for testing
'''

def mpi_skipped_test_mismatching_commsize(mpi_comm, test_name: str, np_needed: int):
  if mpi_comm.Get_rank() == 0:
     print(f"\tWARNING: Test {test_name} skipped for np = {mpi_comm.Get_size()} (requires np = {np_needed})")
