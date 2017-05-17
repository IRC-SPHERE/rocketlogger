#
# Copyright (c) 2017, University of Bristol - http://www.bristol.ac.uk
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice,
#    this list of  conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# Author: Atis Elsts
#
# This file contains Python routines for parsing RocketLogger binary files.
#

import sys
from struct import unpack
from collections import namedtuple
import logging

# Uncomment this to enable debug logging level
#logging.basicConfig(level=logging.DEBUG)

LEAD_IN_SIZE = 56
CHANNEL_SIZE = 28

TIMESTAMP_SIZE = 32

#
# This code support extraction of a single, predefined measurement channel.
#
# The channel used here by default is: Current (I), channel number 1
#
DEFAULT_MEASUREMENT_CHANNEL = "I1"

# -3 for milli-units, -6 for micro-units
RESULT_SCALE = -3

#
# Read the Lead-In of a Rocket Logger file header
#
def read_header_lead_in(f):
    lead_in = f.read(LEAD_IN_SIZE)
    data_format = namedtuple('LeadIn', 'magic version header_len db_size db_count num_samples rate mac_address start_sec start_ns comment_len bc_count ac_count')
    data = data_format._make(unpack('<LHHLLQH6sQQLHH' , lead_in))
    return data

#
# Read binary or analog channel specification in a Rocket Logger file header
#
def read_channel_spec(f):
    channel_spec = f.read(CHANNEL_SIZE)
    data_format = namedtuple('ChannelSpec', 'unit scale size validity_idx name')
    data = data_format._make(unpack('<llHH16s', channel_spec))
    return data

#
# Extracts a single measurement from the RocketLogger file and convert it to RESULT_SCALE
#
class MeasurementExtractor:
    def __init__(self, sample_size, format, low_idx, high_idx, validity_idx, low_scale, high_scale):
        self.sample_size = sample_size
        self.format = format
        self.low_idx = low_idx
        self.high_idx = high_idx
        self.binary_idx = 0
        if validity_idx != 0xffff:
            while validity_idx > 32:
                validity_idx -= 32
                self.binary_idx += 1
            self.validity_bitmask = 1 << (validity_idx - 1)
        else:
            self.validity_bitmask = 0
        # convert from the measurement units to the result units
        self.low_to_milli_scale = 10 ** (low_scale - RESULT_SCALE)
        self.high_to_milli_scale = 10 ** (high_scale - RESULT_SCALE)

    def extract(self, block):
        raw_sample = unpack(self.format, block[:self.sample_size])
        binaries = raw_sample[self.binary_idx]
        # check if the low is valid
        if binaries & self.validity_bitmask:
            measurement_low = raw_sample[self.low_idx]
            return measurement_low * self.low_to_milli_scale, self.sample_size
        # low is not valid, return the high
        measurement_high = raw_sample[self.high_idx]
        return measurement_high * self.high_to_milli_scale, self.sample_size

#
# Parse a data block of a RocketLogger file
#
def parse_block(block, extractor, visitor):
    # TODO: the timestamp is currently ignored!
    pos = TIMESTAMP_SIZE
    while pos < len(block):
        sample, sample_size = extractor.extract(block[pos:])
        pos += sample_size
        visitor.add_sample(sample)

#
# A visitor class that accumulates all samples
#
class DataAccumulator:
    def __init__(self):
        self.data = []
    def add_sample(self, sample):
        self.data.append(sample)
    def get_result(self):
        return self.data

#
# A visitor class that integrates (sums up) all samples
#
class DataAdder:
    def __init__(self):
        self.accumulator = 0
    def add_sample(self, sample):
        self.accumulator += sample
    def get_result(self):
        return self.accumulator


#
# This function processes the header and up to `max_num_blocks` of a RocketLogger file.
# It applies the visitor class to each sample in the data blocks.
#
def operate(filename, visitor, max_num_blocks, measurement_channel=DEFAULT_MEASUREMENT_CHANNEL):
    logging.debug("Opening file " + filename)
    with open(filename, "rb") as f:
        try:
            h = read_header_lead_in(f)
            logging.debug(h)

            if h.comment_len:
                comment = f.read(h.comment_len)
                logging.debug("Comment in the file: " + comment.decode("utf-8"))

            for i in range(h.bc_count):
                binary_channel = read_channel_spec(f)
                logging.debug("Unit: {} scale: {} size: {} valid_idx: {} name: {}".format(
                    binary_channel.unit, binary_channel.scale, binary_channel.size,
                    binary_channel.validity_idx, binary_channel.name.decode("utf-8")))

            # Binary channels are packed effectively
            num_binary_values = (h.bc_count + 31) // 32

            sample_size = 4 * num_binary_values
            low_idx = None
            high_idx = None
            validity_idx = None
            low_scale = None
            high_scale = None
            format = "<"
            for i in range(num_binary_values):
                format += "L"

            for i in range(h.ac_count):
                analog_channel = read_channel_spec(f)
                logging.debug("{} Unit: {} scale: {} size: {} valid_idx: {} name: {}".format(i,
                              analog_channel.unit, analog_channel.scale, analog_channel.size,
                              analog_channel.validity_idx, analog_channel.name.decode("utf-8")))
                sample_size += analog_channel.size
                if analog_channel.size == 4:
                    format += "l"
                elif analog_channel.size == 2:
                    format += "h"
                else:
                    raise Exception("Unsupported chanel size: {} bytes".format(analog_channel.size))
                name = analog_channel.name.decode("utf-8")
                if name.startswith(measurement_channel):
                    if name[len(measurement_channel)] == "H":
                        high_idx = num_binary_values + i
                        high_scale = analog_channel.scale
                    elif name[len(measurement_channel)] == "L":
                        low_idx = num_binary_values + i
                        low_scale = analog_channel.scale
                        validity_idx = analog_channel.validity_idx

            num_blocks = h.db_count
            if max_num_blocks is not None:
                num_blocks = min(num_blocks, max_num_blocks)

            if low_idx is None or high_idx is None or validity_idx is None:
                raise Exception("Unsupported file format: {}".format(h))

            extractor = MeasurementExtractor(sample_size, format, low_idx, high_idx, validity_idx, low_scale, high_scale)
            logging.debug("Processsing {} blocks".format(num_blocks))
            for i in range(num_blocks):
                block = f.read(TIMESTAMP_SIZE + sample_size * h.db_size)
                parse_block(block, extractor, visitor)

        except Exception as ex:
            logging.error("Exception: {}".format(ex))
            return None
    return visitor.get_result()
    

#
# Integrate RocketLogger samples. Returns a scalar value: the sum of the samples.
#
def integrate(filename, max_num_blocks=None, measurement_channel=DEFAULT_MEASUREMENT_CHANNEL):
    d = DataAdder()
    return operate(filename, d, max_num_blocks, measurement_channel)

#
# Load RocketLogger samples. Returns a list of all the samples.
#
def load(filename, max_num_blocks=None, measurement_channel=DEFAULT_MEASUREMENT_CHANNEL):
    d = DataAccumulator()
    return operate(filename, d, max_num_blocks, measurement_channel)

###########################################

#
# A simple usage demo
#
def main():
    if len(sys.argv) > 1:
        RLD_FILE_NAME = sys.argv[1]
    else:
        RLD_FILE_NAME = "test.rld"
    integral = integrate(RLD_FILE_NAME, max_num_blocks=3)
    print("Cumulative charge: ", integral)
    values = load(RLD_FILE_NAME, max_num_blocks=3)
    print("Values: ", values[:10])
    print("Cumulative charge calculated in an alternative way: ", sum(values))

###########################################

if __name__ == '__main__':
    main()
