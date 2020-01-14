#!/usr/bin/env python

class CSVLogger:

    def __init__(self, save_path, fields):
        self.save_path = save_path
        self.f = open(save_path, 'w')
        self.f.write(','.join(fields)+'\n')

    def log(self, *args):
        sargs = [ str(arg) for arg in args ]
        self.f.write(','.join(sargs)+'\n')
        self.f.flush()

    def close():
        self.f.close()
