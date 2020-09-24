import os
import io
import numpy as np
import torch

try:
    from openvino.inference_engine import IECore, StatusCode
    from .loader import convert_to_2d
except ImportError:
    pass


def load_openvino_model(model, dirname):
    package = model.config['model']['package']
    if package == 'bonito.ctc':
        return OpenVINOCTCModel(model, dirname)
    elif package == 'bonito.crf':
        return OpenVINOCRFModel(model, dirname)
    else:
        raise Exception('Unknown model configuration: ' + package)


class OpenVINOModel:

    def __init__(self, model, dirname):
        self.model = model
        self.alphabet = model.alphabet
        self.parameters = model.parameters
        self.stride = model.stride
        self.net = None
        self.exec_net = None
        self.dirname = dirname
        self.ie = IECore()


    def eval(self):
        pass


    def half(self):
        return self


    @property
    def config(self):
        return self.model.config


    def to(self, device):
        self.device = str(device).upper()

    """
    Call this method once to initialize executable network
    """
    def init_model(self, model, inp_shape):
        # First, we try to check if there is IR on disk. If not - load model in runtime
        xml_path, bin_path = [os.path.join(self.dirname, 'model') + ext for ext in ['.xml', '.bin']]
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            self.net = self.ie.read_network(xml_path, bin_path)
        else:
            # Convert model to ONNX buffer
            buf = io.BytesIO()
            inp = torch.randn(inp_shape)
            torch.onnx.export(model, inp, buf, input_names=['input'], output_names=['output'],
                              opset_version=11)

            # Import network from memory buffer
            self.net = self.ie.read_network(buf.getvalue(), b'', init_from_buffer=True)

        # Load model to device
        config = {}
        if self.device == 'CPU':
            config={'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
        self.exec_net = self.ie.load_network(self.net, self.device,
                                             config=config, num_requests=0)


    def process(self, data):
        data = data.float()
        batch_size = data.shape[0]
        inp_shape = list(data.shape)
        inp_shape[0] = 1  # We will run the batch asynchronously

        # List that maps infer requests to index of processed chunk from batch.
        # -1 means that request has not been started yet.
        infer_request_input_id = [-1] * len(self.exec_net.requests)
        out_shape = self.net.outputs['output'].shape
        # CTC network produces 1xWxNxC
        output = np.zeros([out_shape[-3], batch_size, out_shape[-1]], dtype=np.float32)

        for inp_id in range(batch_size):
            # Get idle infer request
            infer_request_id = self.exec_net.get_idle_request_id()
            if infer_request_id < 0:
                status = self.exec_net.wait(num_requests=1)
                if status != StatusCode.OK:
                    raise Exception("Wait for idle request failed!")
                infer_request_id = self.exec_net.get_idle_request_id()
                if infer_request_id < 0:
                    raise Exception("Invalid request id!")

            out_id = infer_request_input_id[infer_request_id]
            request = self.exec_net.requests[infer_request_id]

            # Copy output prediction
            if out_id != -1:
                output[:,out_id:out_id+1] = request.output_blobs['output'].buffer

            # Start this request on new data
            infer_request_input_id[infer_request_id] = inp_id
            request.async_infer({'input': data[inp_id]})
            inp_id += 1

        # Wait for the rest of requests
        status = self.exec_net.wait()
        if status != StatusCode.OK:
            raise Exception("Wait for idle request failed!")
        for infer_request_id, out_id in enumerate(infer_request_input_id):
            if out_id == -1:
                continue
            request = self.exec_net.requests[infer_request_id]
            output[:,out_id:out_id+1] = request.output_blobs['output'].buffer

        return torch.tensor(output)


class OpenVINOCTCModel(OpenVINOModel):

    def __init__(self, model, dirname):
        super().__init__(model, dirname)


    def __call__(self, data):
        data = data.unsqueeze(2)  # 1D->2D
        if self.exec_net is None:
            convert_to_2d(self.model)
            self.init_model(self.model, [1, 1, 1, data.shape[-1]])

        return self.process(data)


    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        return self.model.decode(x, beamsize=beamsize, threshold=threshold,
                                 qscores=qscores, return_path=return_path)


class OpenVINOCRFModel(OpenVINOModel):

    def __init__(self, model, dirname):
        super().__init__(model, dirname)
        self.seqdist = model.seqdist


    def __call__(self, data):
        if self.exec_net is None:
            self.init_model(self.model.encoder, [1, 1, data.shape[-1]])

        return self.process(data)


    def decode(self, x):
        return self.model.decode(x)


    def decode_batch(self, x):
        return self.model.decode_batch(x)
