# Dense Captioning Events in Video - Evaluation Code
This is a forked repo from the official 2017 [densevid_eval](https://github.com/ranjaykrishna/densevid_eval/tree/b8d90707984bf9c99454ba82b089006f14fb62b3). You can also find the latest official 2018 repo [here](https://github.com/ranjaykrishna/densevid_eval) and the one with SPICE support [here](https://github.com/LuoweiZhou/densevid_eval_spice).

## Usage
First, clone this repository and make sure that all the submodules are also cloned properly.
```
git clone --recursive https://github.com/LuoweiZhou/densevid_eval.git
```

Next, download the dataset using
```
./download.sh
```

Finally, test that the evaluator runs by testing it on our ```sample_submission.json``` file by calling:
```
python evaluate.py
```

You are now all set to produce your own dense captioning results for videos and use this code to evaluate your mode:
```
python evaluate.py -s YOUR_SUBMISSION_FILE.JSON
```

## Paper
Visit [here](http://cs.stanford.edu/people/ranjaykrishna/densevid) for details.

## Citation
```
@inproceedings{krishna2017dense,
    title={Dense-Captioning Events in Videos},
    author={Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Niebles, Juan Carlos},
    booktitle={ArXiv},
    year={2017}
}
```

## License

MIT License copyright Ranjay Krishna
