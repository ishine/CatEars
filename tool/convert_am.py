import sys
import re
import struct
import math
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python3 {}: <text-nnet2-am> <am-bin>".format(sys.argv[0]))
    print("Convert Kaldi nnet2 AM to pocketkaldi format.")
    print("    text-nnet2-am: The text format of nnet2 AM file could be obtained by kaldi/src/nnet2bin/nnet-am-copy.")
    sys.exit(1)

from_file = sys.argv[1]
to_file = sys.argv[2]


# Ids for different layers
LINEAR_LAYER = 0
RELU_LAYER = 1
SPLICE_LAYER = 6
BATCHNORM_LAYER = 7
LOGSOFTMAX_LAYER = 8

class Layer:
    def write_vector(self, fd, vec):
        fd.write(b"VEC0")
        fd.write(struct.pack("<i", len(vec) * 4 + 4))
        fd.write(struct.pack("<i", len(vec)))
        for v in vec:
            fd.write(struct.pack("<f", v))

    def write_matrix(self, fd, mat):
        fd.write(b"MAT0")
        fd.write(struct.pack("<i", 8))
        fd.write(struct.pack("<i", len(mat)))
        fd.write(struct.pack("<i", len(mat[0])))
        for col in mat:
            self.write_vector(fd, col)
    
    def write(self, fd):
        fd.write(b"LAY0")
        fd.write(struct.pack("<i", self.layer_type))

    def output_dim(self, input_dim):
        return input_dim
    
    def input_dim(self):
        return None

    def __str__(self):
        return self.layer_name

class LinearLayer(Layer):
    def __init__(self, W, b):
        assert(len(W.shape) == 2 and len(b.shape) == 1)
        self.W = W
        self.b = b
        self.layer_name = 'LinearLayer'
        self.layer_type = LINEAR_LAYER
    
    def write(self, fd):
        super().write(fd)
        self.write_matrix(fd, self.W)
        self.write_vector(fd, self.b)
    
    def input_dim(self):
        return self.W.shape[0]
    
    def output_dim(self, input_dim):
        return self.W.shape[1]
    
    def __str__(self):
        return "{}: W = ({}, {}), b = ({})".format(
            self.layer_name, self.W.shape[0], self.W.shape[1], self.b.shape[0])

class ReluLayer(Layer):
    def __init__(self):
        self.layer_name = 'ReluLayer'
        self.layer_type = RELU_LAYER

class SpliceLayer(Layer):
    def __init__(self, indices):
        assert(len(indices) > 0)
        self.layer_name = 'SpliceLayer'
        self.layer_type = SPLICE_LAYER
        self.indices = indices

    def output_dim(self, input_dim):
        if input_dim == None:
            return None
        return input_dim * len(self.indices)

    def write(self, fd):
        super().write(fd)
        fd.write(struct.pack("<i", len(self.indices)))
        for idx in self.indices:
            fd.write(struct.pack("<i", idx))

    def __str__(self):
        return "{}: indices = {}".format(self.layer_name, self.indices)

class BatchNormLayer(Layer):
    def __init__(self, eps):
        self.layer_name = 'BatchNormLayer'
        self.layer_type = BATCHNORM_LAYER
        self.eps = eps

    def write(self, fd):
        super().write(fd)
        fd.write(struct.pack("<f", self.eps))

    def __str__(self):
        return "{}: eps = {}".format(self.layer_name, self.eps)

class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.layer_name = 'LogSoftmaxLayer'
        self.layer_type = LOGSOFTMAX_LAYER

class AM:
    def __init__(self, layers):
        self.left_context = 0
        self.right_context = 0
        self.prior = None
        self.layers = layers

    def verify(self):
        output_dim = None
        for idx, layer in enumerate(self.layers):
            expected_dim = layer.input_dim()
            if output_dim != None and expected_dim != None and output_dim != expected_dim:
                raise Exception('input_dim == {} expected, but {} found in layer {}'.format(
                    expected_dim,
                    output_dim,
                    idx))
            output_dim = layer.output_dim(output_dim)

    def write(self, filename):
        with open(filename + ".nnet", 'wb') as fd:
            fd.write(b"NN01")
            fd.write(struct.pack("<i", len(self.layers)))
            for layer in self.layers:
                layer.write(fd)
        with open(filename + ".prior", 'wb') as fd:
            Layer().write_vector(fd, self.prior)

re_tag = re.compile(r'<(.*?)>(.*?)</(.*?)>', re.DOTALL)

def goto_token(token_name, text):
    re_token = re.compile(r'<{}>(.*)'.format(token_name), re.DOTALL)
    m = re_token.search(text)
    if m == None:
        raise Exception('unable to find token: {}'.format(token_name))
    return m.group(1)

def read_until_token(token_name, text):
    re_token = re.compile(r'(.*?)<{}>'.format(token_name), re.DOTALL)
    m = re_token.search(text)
    if m == None:
        raise Exception('unable to find token: {}'.format(token_name))
    return m.group(1)

def read_string(text):
    m = re.search(r'^\s*([-_A-Za-z0-9\.]+)\s+(.*)', text, re.DOTALL)
    if m == None:
        raise Exception('read_string failed')
    return (m.group(1), m.group(2))

def read_token(text):
    m = re.search(r'^\s*<(.*?)>(.*)', text, re.DOTALL)
    if m == None:
        raise Exception('read_token failed')
    return (m.group(1), m.group(2))

def read_int(text):
    m = re.search(r'^\s*(\d+)\s+(.*)', text, re.DOTALL)
    if m == None:
        raise Exception('read_int failed')
    return (int(m.group(1)), m.group(2))

def read_float(text):
    m = re.search(r'^\s*((?:-?\d+)(?:\.(?:\d+))?(?:e-?\d+)?)\s+(.*)', text, re.DOTALL)
    if m == None:
        raise Exception('read_float failed')
    return (float(m.group(1)), m.group(2))

def read_matrix(text, num_type = float):
    m = re.search(r'^\s*\[(.*?)\]\s*(.*)', text, re.DOTALL)
    if m == None:
        raise Exception('read_matrix failed')
    remained = m.group(2)
    text = m.group(1)
    lines = text.split('\n')
    matrix_cols = []
    row_num = 0
    for line in lines:
        if line.strip() == '': continue
        matrix_cols.append(list(map(num_type, line.strip().split())))
        if row_num == 0:
            row_num = len(matrix_cols[0])
        elif row_num != len(matrix_cols[-1]):
            raise Exception('Row number mismatch')
    return np.array(matrix_cols), remained

re_component = re.compile(r'^component-node name=(.*?) component=(.*?) input=(.*?)$')
re_input = re.compile(r'^Append\((.*)\)$')
re_split = re.compile(r'(Offset\([\w\.]+, *-?\d+\)|[\w\.]+)')
re_offset = re.compile(r'^Offset\(([\w\.]+), *(-?\d+)\)$')
def parse_nnet3_desc(desc_text):
    lines = desc_text.split('\n')
    prev_name = 'input'
    layers = []
    layer_dict = {}
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        node_type = line.split()[0]
        assert(node_type in {'component-node', 'input-node', 'output-node'})
        if node_type == 'component-node':
            m = re_component.match(line)
            assert(m != None)
            layer_input = m.group(3).strip()
            layer_comp = m.group(2)
            m_input = re_input.match(layer_input)
            if m_input != None:
                indices = []
                fields = re_split.split(m_input.group(1))
                for field in fields:
                    m_offset = re_offset.match(field.strip())
                    if m_offset:
                        from_comp = m_offset.group(1)
                        index = int(m_offset.group(2))
                        assert(from_comp == prev_name)
                        indices.append(index)
                    elif field.strip() in {',', ''}:
                        pass
                    else:
                        assert(field.strip() == prev_name)
                        indices.append(0)
                layer_name = layer_comp + '_splice'
                layer_dict[layer_name] = SpliceLayer(indices)
                layers.append(layer_name)
            else:
                assert(layer_input == prev_name)
            layers.append(layer_comp)
            prev_name = layer_comp
    return layers, layer_dict

# Nnet token
def read_nnet(model_text):
    remained_text = goto_token('Nnet3', model_text)
    remained_text = read_until_token('/Nnet3', remained_text)
    nnet3_desc = read_until_token('NumComponents', remained_text)

    remained_text = goto_token('NumComponents', remained_text)
    num_components, remained_text = read_int(remained_text)
    print('num_components = {}'.format(num_components))
    print('------------------ nnet3_desc ------------------')
    print(nnet3_desc)
    layers, layer_dict = parse_nnet3_desc(nnet3_desc)

    # Tokens in Components
    print('------------------ read_layer ------------------')
    while remained_text.strip() != '':
        comp_name_tag, remained_text = read_token(remained_text)
        assert(comp_name_tag == 'ComponentName')
        comp_name, remained_text = read_string(remained_text)
        token_tag, remained_text = read_token(remained_text)
        print(comp_name, '=', token_tag)
        end_tag = '/' + token_tag
        content_text = read_until_token(end_tag, remained_text)
        remained_text = goto_token(end_tag, remained_text)

        # Parse token_text
        if token_tag == 'NaturalGradientAffineComponent':
            content_text = goto_token('LinearParams', content_text)
            W, content_text = read_matrix(content_text)
            content_text = goto_token('BiasParams', content_text)
            b, content_text = read_matrix(content_text)
            layer_dict[comp_name] = LinearLayer(W.T, b[0])
        elif token_tag == 'RectifiedLinearComponent':
            layer_dict[comp_name] = ReluLayer()
        elif token_tag == 'BatchNormComponent':
            content_text = goto_token('Epsilon', content_text)
            eps, _ = read_float(content_text)
            layer_dict[comp_name] = BatchNormLayer(eps)
        elif token_tag == 'LogSoftmaxComponent':
            layer_dict[comp_name] = LogSoftmaxLayer()
        else:
            raise Exception('unexpected layer name: ' + token_tag)
        print(str(layer_dict[comp_name]))

    print('------------------ layers ------------------')
    layer_objects = []
    for i, layer_name in enumerate(layers):
        if layer_name in layer_dict:
            layer = layer_dict[layer_name]
            layer_objects.append(layer)
            print('layer {}: {}'.format(i, str(layer)))
        else:
            raise Exception('layer not found: ' + layer_name)
    
    return layer_objects

if __name__ == '__main__':
    with open(from_file) as fd:
        model_text = fd.read()

    layers = read_nnet(model_text)
    am = AM(layers)

    # Prior
    remained_text = goto_token('Priors', model_text)
    prior, content_text = read_matrix(remained_text)
    print('------------------ prior ------------------')
    print('Prior: {} * {}'.format(len(prior), len(prior[0])))
    am.prior = prior[0]

    am.verify()
    am.write(to_file)
