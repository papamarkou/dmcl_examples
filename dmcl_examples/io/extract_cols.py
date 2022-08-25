import pandas as pd

def extract_cols_from_csv(
    in_path, out_path, cols, header=None, dtype='str', mode='a', chunk_size=1000, verbose=True, num_lines=None
):
    out_head = False if (header is None) else True

    if verbose:
        if num_lines is None:
            num_lines = len(pd.read_csv(in_path, header=header, dtype=dtype))

        num_chunks, mod = divmod(num_lines, chunk_size)
        if mod != 0:
            num_chunks = num_chunks + 1

        msg = 'Extracting columns from chunk {:' + str(len(str(num_chunks))) + '} out of ' + str(num_chunks) + '...'

    for i, chunk in enumerate(pd.read_csv(in_path, usecols=cols, header=header, dtype=dtype, chunksize=chunk_size)):
        if verbose:
            print(msg.format(i+1))

        chunk.to_csv(out_path, index=None, header=out_head if (i == 0) else False, mode=mode)
