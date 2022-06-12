import copy
import time
import multiprocessing

import pandas as pd
import numpy as np

# only even numbers pls
FRAME_SIZE = 1000000

TESTFRAME = pd.DataFrame({'a': [1] * (FRAME_SIZE // 2) + [2] * (FRAME_SIZE // 2),
                          'b': [2] * (FRAME_SIZE // 2) + [1] * (FRAME_SIZE // 2),
                          'c': [0] * FRAME_SIZE})


def do_math(x, y):
    return (x + y) * 100


v_do_math = np.vectorize(do_math, otypes=[float])


def frame_display(inner):
    def wrapper(*args, **kwargs):
        title = kwargs.pop('title')
        print(title)
        start = time.time()
        frame = inner(*args, **kwargs)
        print(f'Calculation time: {time.time() - start}')
        print(frame.head(n=10).to_string())
        print('\n\n')

    return wrapper


def calc_on_frame(frame, start, end, return_frame=False):
    a = frame.iloc[start:end].loc[:, 'a'].values
    b = frame.iloc[start:end].loc[:, 'b'].values
    c = v_do_math(a, b)

    frame['c'].iloc[start:end] = c

    if return_frame:
        return frame


@frame_display
def standard():
    frame = copy.deepcopy(TESTFRAME)
    calc_on_frame(frame, 0, FRAME_SIZE)
    return frame


@frame_display
def split_multiprocess():
    frame = copy.deepcopy(TESTFRAME)
    frame1 = frame.iloc[0:FRAME_SIZE // 2, :]
    frame2 = frame.iloc[FRAME_SIZE // 2:FRAME_SIZE, :]
    with multiprocessing.Pool() as pool:
        # You need to the return_frame arg in calc_on_frame to be true for this to work
        frames = pool.starmap(calc_on_frame, [[frame1, 0, FRAME_SIZE // 2, True], [frame2, 0, FRAME_SIZE // 2, True]])

    return pd.concat(frames)


@frame_display
def inplace():

    frame = copy.deepcopy(TESTFRAME)
    # idea: Create multiple references to the frame's location in memory
    frame1 = frame
    frame2 = frame

    # This doesn't work! Pass by reference doesn't happen in pool.starmap - copies of the objects are made
    with multiprocessing.Pool() as pool:
        pool.starmap(calc_on_frame, [[frame1, 0, FRAME_SIZE // 2], [frame2, FRAME_SIZE // 2, FRAME_SIZE]])

    # This object is unaltered, as a result
    return frame


if __name__ == '__main__':
    standard(title="Standard Frame Calculation")

    split_multiprocess(title="Split Multiprocess Calculation")

    inplace(title="Inplace")

