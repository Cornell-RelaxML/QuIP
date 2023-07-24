import time
from method import QuantMethod


class Nearest(QuantMethod):

    def fasterquant(self):
        tick = time.time()
        full_W = self.layer.weight.data.clone()
        W      = self.layer.weight.data.clone()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        # self.layer.weight.data = quantize(
        #     W, quantizer.scale, quantizer.zero,
        #     quantizer.maxq).to(self.layer.weight.data.dtype)
        self.layer.weight.data = self.quantizer.quantize(W).to(self.layer.weight.data.dtype)
        self.postproc()
        # print('time %.2f' % (time.time() - tick))
        self.time = time.time() - tick
        self.error_compute(full_W, self.layer.weight.data)
