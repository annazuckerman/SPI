import unittest
import random
import numpy as np
import os
import pandas as pd
import sys
import h5py
import shutil
sys.path.append('../')
import utils  # nopep8


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        """ Set up for unit testing by creating toy data
        """
        cls.vlsvpath = 'example_data/state00001250.vlsv'
        cls.outdir = './test_outdir'
        cls.outfile = 'test_outfile.h5'

    @classmethod
    def tearDownClass(cls):

        """ Tear down unit testing toy data
        """
        cls.testvlsv = None
        cls.outdir = None
        cls.outfile = None
        shutil.rmtree('./test_outdir')

    def test_vlsv_to_h5(cls):

        """ Unit tests for vlsv_to_h5() function
        """
        utils.vlsv_to_h5(cls.vlsvpath, cls.outdir, cls.outfile)

        # positive tests:
        # check that file exists
        self.assertTrue(os.path.exists(cls.outdir + '/' + cls.outfile))
        # check that data has the right dimenstionality
        f = h5py.File(cls.outdir + '/' + cls.outfile, 'r')
        self.assertTrue(len(np.shape(np.array(f['magnetic_field_x']))) == 3)

        # negative tests:
        # check that the fields are not empty
        self.assertFalse(len(f.keys()) == 0)

        # error raising tests:
        self.assertRaises(TypeError, utils.vlsv_to_h5, 1, 2, 3)
        self.assertRaises(FileNotFoundError,  utils.vlsv_to_h5,
                          './inputpath', cls.outdir, cls.outfile)

    #def test_plot_3d(cls):
        
