import os
import shutil
from multiprocessing import Manager, Process, Queue

from gigapath.pipeline import tile_one_slide

DATA_ROOT = "/opt/gpudata/skin-cancer"


def tile(id: int, q: Queue, dest_root: str, overwrite: bool = False) -> None:
    """
    Removes tiling jobs from the queue and creates tiles using gigapath's
    specifications.

    Parameters
    ----------
    id : int
        The process id

    q : multiprocessing.Queue
        A queue containing file paths with WSIs

    dest_root : str
        The destination root directory for the output tiles

    overwrite : bool
        Whether to overwrite existing tiles
    """
    output_dir = os.path.join(dest_root, "output")
    thumbnail_dir = os.path.join(dest_root, "thumbnails")

    while True:
        # pop from the queue; if sentinel reached, break out of loop
        slide_path = q.get()
        if slide_path == "DONE":
            break

        slide_name = os.path.basename(slide_path)
        tiles_dir = os.path.join(output_dir, slide_name)
        exists = os.path.exists(tiles_dir)

        # tile if condition met
        if not exists or overwrite:
            # if exists, needs overwrite so remove existing tiles and
            # associated files before tiling
            if exists:
                shutil.rmtree(tiles_dir)
                os.remove(
                    os.path.join(thumbnail_dir, slide_name + "_original.png")
                )
                os.remove(
                    os.path.join(thumbnail_dir, slide_name + "_roi_tiles.png")
                )
                os.remove(os.path.join(thumbnail_dir, slide_name + "roi.png"))

            # then tile the slide
            tile_one_slide(slide_path, dest_root)

    # once tasks complete, print complete and exit
    print(f"worker {id} finished", flush=True)


def main():
    """
    Populates the job queue with file paths and spawns processes to pull from
    the queue.
    """
    m = Manager()
    q = (
        m.Queue()
    )  # not a joinablequeue to avoid processes being killed prematurely
    num_threads = 12

    src_root = os.path.join(DATA_ROOT, "data/slides")
    dest_root = os.path.join(DATA_ROOT, "data/tiles")

    # add all the filenames to the queue + sentinels
    for dname, _, fnames in os.walk(src_root):
        for fname in fnames:
            q.put(os.path.join(dname, fname))
    for _ in range(num_threads):
        q.put("DONE")

    # create pool of processes and start them
    processes = []
    for i in range(num_threads):
        proc = Process(target=tile, args=(i, q, dest_root, False))
        processes.append(proc)
        print(f"starting process {i}")
        proc.start()

    # wait for all processes to finish
    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()
