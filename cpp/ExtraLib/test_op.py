# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import shutil
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import math

LIB_NAME = 'extra_lib'

def load_op_module(lib_name):
  """
  Load TensorFlow operator library.
  """
  # use absolute path so that ops.py can be called from other directory
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'build/lib{0}.so'.format(lib_name))
  # duplicate library with a random new name so that
  # a running program will not be interrupted when the original library is updated
  lib_copy_path = '/tmp/lib{0}_{1}.so'.format(str(uuid.uuid4())[:8], LIB_NAME)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  #print(_)
  return oplib

op_module = load_op_module(LIB_NAME)

overlaps = [[0.1, 0.4, 0.6, 0.2, 0.7], [0.5, 0.14, 0.76, 0.32, 0.47], [0.21, 0.94, 0.66, 0.22, 0.57], [0.91, 0.14, 0.26, 0.42, 0.67], [0.11, 0.84, 0.26, 0.42, 0.57]]
ops.NotDifferentiable("SmallMiningMatch")
ops.NotDifferentiable("DynamicAnchorRouting")

class SmallMiningMatchTest(tf.test.TestCase):
  def testSmallMiningMatch(self):
    # with tf.device('/gpu:1'):
    #   # map C++ operators to python objects
    #   small_mining_match = op_module.small_mining_match
    #   result = small_mining_match(overlaps, 0., 0.5, 0.5, 2, 0.1)
    #   with self.test_session() as sess:
    #     print('small_mining_match in gpu:', sess.run(result))
    with tf.device('/cpu:0'):
      # map C++ operators to python objects
      small_mining_match = op_module.small_mining_match
      result = small_mining_match(overlaps, 0., 0.6, 0.6, 5, 0.1)
      with self.test_session() as sess:
        print('small_mining_match in cpu:', sess.run(result))


class DynamicAnchorRoutingTest(tf.test.TestCase):
  def testDynamicAnchorRouting(self):
    # with tf.device('/gpu:1'):
    #   # map C++ operators to python objects
    #   dynamic_anchor_routing = op_module.dynamic_anchor_routing
    #   result = dynamic_anchor_routing(overlaps, 0., 0.5, 0.5, 2, 0.1)
    #   with self.test_session() as sess:
    #     print('dynamic_anchor_routing in gpu:', sess.run(result))
    with tf.device('/cpu:0'):
      # map C++ operators to python objects
      dynamic_anchor_routing = op_module.dynamic_anchor_routing
      result = dynamic_anchor_routing([[100., 120, 310, 420]], [[120., 110, 330, 400]], [1.], [1], 600, 600, 1, 1, True)
      with self.test_session() as sess:
        print('dynamic_anchor_routing in cpu:', sess.run(result))

if __name__ == "__main__":
  tf.test.main()

