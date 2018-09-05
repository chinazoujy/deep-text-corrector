from flask import Flask, request
import tensorflow as tf
from correct_text import create_model, DefaultMovieDialogConfig, decode
from text_corrector_data_readers import MovieDialogReader

data_path = '/input/data/movie_dialog_train.txt'
model_path = '/input/model'

tfs = tf.Session()

config = DefaultMovieDialogConfig()
print('Loading model from path: %s' % model_path)
model = create_model(tfs, True, model_path, config=config)
print('Using data from path: %s' % data_path)
data_reader = MovieDialogReader(config, data_path)

app = Flask(__name__)


@app.route('/', methods=['POST'])
def correct_handler():
    corrective_tokens = data_reader.read_tokens(data_path)
    request.get_data()
    decodings = decode(tfs, model=model, data_reader=data_reader,
                       data_to_decode=[request.data.split()],
                       corrective_tokens=corrective_tokens)
    return ' '.join(next(decodings))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
