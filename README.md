# RocketLogger

[RocketLogger](https://git.ee.ethz.ch/sigristl/rocketlogger) binary .rld file parser in Python.

Command line usage:

    pypy ./rldfile.py filename.rld

or 

    python ./rldfile.py filename.rld

where `filename.rld` is a valid RocketLogger binary file.

Usage as a Python module:

    import rocketlogger

    SAMPLING_RATE = 16000
    RLD_FILE_NAME = "test.rld"
    integral = rocketlogger.integrate(RLD_FILE_NAME, max_num_blocks=3, measurement_channel="I1")
    print("Cumulative charge, mC: ", integral / SAMPLING_RATE)
    values = rocketlogger.load(RLD_FILE_NAME, max_num_blocks=3, measurement_channel="I1")
    print("Values: ", values[:10])
    print("Cumulative charge in mC, calculated in an alternative way: ", sum(values) / SAMPLING_RATE)

The module exports two functions:

* `integrate` - sums up the contents of the measurements and returns the sum. Divide the returned value to obtain the total charge in millicoulombs (mC).
* `load` - returns the list of current or voltage measurements, in milliamps (mA) or millivolts (mV) respectively.

PyPy is the recommended Python interpreter (instead of CPython). Mostly because of this reason the module does not use NumPy in any way.
