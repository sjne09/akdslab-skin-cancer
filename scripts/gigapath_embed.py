import gc
import os
import pickle
from typing import Dict, List, Tuple

import torch
from gigapath.pipeline import load_tile_slide_encoder, run_inference_with_tile_encoder
from torch import nn
from torch.multiprocessing import Manager, Process, Queue, set_start_method

DATA_ROOT = "/opt/gpudata/skin-cancer"


def create_tile_embeds(
    tile_encoder_model: nn.Module, tiles_dir: str, num_gpus: int
) -> Queue:
    """
    Create tile embeddings for each slide in the provided tiles directory.

    Parameters
    ----------
    tile_encoder_model : nn.Module
        The tile encoder

    tiles_dir : str
        The directory containing subdirectories with tiles. Each subdirectory
        corresponds to a single slide

    num_gpus : int
        The number of GPUs to run on

    Returns
    -------
    torch.multiprocessing.Queue[Tuple[str, Dict[str, torch.Tensor]]]
        A job queue containing tuples of the slide names and a dict containing
        tile encoder outputs with key 'tile_embeds' and tile coordinates with key
        'coords'
    """
    m = Manager()
    tile_embeds: Queue[Tuple[str, Dict[str, torch.Tensor]]] = m.Queue()

    for dirname in os.listdir(tiles_dir):
        dpath = os.path.join(tiles_dir, dirname)
        if os.path.isdir(dpath):
            tiles = [
                os.path.join(dpath, fname)
                for fname in os.listdir(dpath)
                if fname.endswith(".png")
            ]
            if len(tiles) > 0:
                print(f"running for {dpath} with {len(tiles)} tiles")

                # inference method will move model and tiles to cuda device; model return
                # includes tensor containing embeddings and tensor containing tile coords
                embeds = (
                    dirname,
                    run_inference_with_tile_encoder(
                        tiles, tile_encoder_model, batch_size=128 * num_gpus
                    ),
                )
                tile_embeds.put(embeds)

                # pickle the embeddings in case of error down the line
                with open(
                    os.path.join(
                        DATA_ROOT,
                        f"outputs/tile_embeddings/{os.path.splitext(dirname)[0]}.pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(embeds[1], f)

    return tile_embeds


def create_slide_embeds(
    id: int,
    q: Queue,
    outputs: Dict[str, torch.Tensor],
    device: torch.device,
    slide_encoder_model: nn.Module,
) -> None:
    """
    Create the slide embeddings by removing jobs from the queue and running
    inference on each slide individually. Running one slide at a time is necessary
    due to tiles per slide being heterogeneous.

    Parameters
    ----------
    id : int
        ID for the worker process

    q : torch.multiprocessing.Queue[Tuple[str, Dict[str, torch.Tensor]]]
        A job queue containing tuples of the slide names and a dict containing
        tile encoder outputs with key 'tile_embeds' and tile coordinates with key
        'coords'

    outputs : Dict[str, torch.Tensor]
        A shared dict between worker processes to add slide embeddings to. Keys are
        slide names and values are the embedding tensors

    device : torch.device
        The device to use

    slide_encoder_model : nn.Module
        The slide encoder
    """
    while True:
        tile_data = q.get()
        if tile_data == "DONE":
            break
        slide_name = tile_data[0]
        print(f"running for {slide_name}", flush=True)
        tile_embeds: torch.Tensor = tile_data[1]["tile_embeds"]
        coords: torch.Tensor = tile_data[1]["coords"]
        slide_embed = run_slide_inference(
            tile_embeds, coords, device, slide_encoder_model
        )
        outputs[slide_name] = slide_embed

    print(f"worker {id} finished", flush=True)


def run_slide_inference(
    tile_embeds: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
    slide_encoder_model: nn.Module,
) -> torch.Tensor:
    """
    Run inference using the slide encoder.

    Parameters
    ----------
    tile_embeds : torch.Tensor
        The tile embeddings; shape (N, H) or (B, N, H) where N is the number of
        tiles for the slide, H is the embedding dims, and B is the batch size

    coords : torch.Tensor
        The slide coords corresponding to the tile embeddings; shape (N, 2) or
        (B, N, 2) where N is the number of tiles for the slide, and B is the batch
        size

    device : torch.device
        The device to use

    slide_encoder_model : nn.Module
        The slide encoder

    Returns
    -------
    torch.Tensor
        The output from the final layer of the model; shape (B, H_out)
    """
    # add batch dim to embedding tensor if necessary
    if len(tile_embeds.shape) == 2:
        tile_embeds = tile_embeds.unsqueeze(0)
        coords = coords.unsqueeze(0)

    slide_encoder_model = slide_encoder_model.to(device)
    slide_encoder_model.eval()

    # run inference
    tile_embeds = tile_embeds.to(device)
    coords = coords.to(device)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        slide_embeds = slide_encoder_model(tile_embeds, coords, all_layer_embed=True)
    outputs = {
        "layer_{}_embed".format(i): slide_embeds[i].detach().cpu()
        for i in range(len(slide_embeds))
    }
    outputs["last_layer_embed"] = slide_embeds[-1].detach().cpu()

    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # return only the final embedding
    return outputs["last_layer_embed"]


def main():
    gpus = ["0", "1"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_processes = len(gpus)
    print(f"Using device {device}")

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    # load models
    tile_enc: nn.Module
    slide_enc: nn.Module
    tile_enc, slide_enc = load_tile_slide_encoder(global_pool=True)
    tile_enc = nn.DataParallel(tile_enc)
    print("---")

    # get tile embeddings in job queue
    tile_embeds = create_tile_embeds(
        tile_enc, os.path.join(DATA_ROOT, "data/tiles/output"), num_processes
    )
    print("---")

    # remove tile encoder from device to free memory
    del tile_enc
    gc.collect()
    torch.cuda.empty_cache()

    # add sentinels to tile embeds queue
    for _ in range(num_processes):
        tile_embeds.put("DONE")

    # run multiple slide encoders in parallel
    procs: List[Process] = []
    m = Manager()
    slide_embeds: Dict[str, torch.Tensor] = m.dict()
    for i, gpu in enumerate(gpus):
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
        print(f"using device {device}")
        proc = Process(
            target=create_slide_embeds,
            args=(i, tile_embeds, slide_embeds, device, slide_enc),
        )
        procs.append(proc)
        proc.start()

    # wait until the queue is empty
    for proc in procs:
        proc.join()

    # convert slide embeds to regular dict to allow pickling
    slide_embeds = dict(slide_embeds)

    # pickle the output embeddings
    with open(os.path.join(DATA_ROOT, "outputs/gigapath_slide_embeds.pkl"), "wb") as f:
        pickle.dump(slide_embeds, f)


if __name__ == "__main__":
    main()
