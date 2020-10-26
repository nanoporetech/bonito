import numpy as np
from kbeam import beamsearch

n_bases = 4
beam_cut = 100.0
beamsizes = (5, 40)
data_dir = "/media/groups/platform/temp/kbeam-testdata"

scores = np.load("%s/scores.npy" % data_dir) # (20, 960, 5120) N, T, C
guides = np.load("%s/guides.npy" % data_dir) # (20, 961, 1024) N, T + 1, C

for beamsize in beamsizes:

    targets = np.load("%s/states_beamsize_%s.npy" % (data_dir, beamsize))
    targlen = np.load("%s/states_lens_beamsize_%s.npy" % (data_dir, beamsize))

    for score, guide, target, target_len in zip(scores, guides, targets, targlen):
        states, _ = beamsearch(score, n_bases, beamsize, guide=guide, beam_cut=beam_cut)
        np.testing.assert_array_equal(states, target[:target_len])
        print(".", end="")

    print("\nBEAMSIZE %s: [PASSED]" % beamsize)
