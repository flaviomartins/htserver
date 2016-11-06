from __future__ import print_function, unicode_literals, division
import json
import logging
from wsgiref import simple_server

import falcon

import plac
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


logger = logging.getLogger(__name__)


TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
STOPWORDS = set(stopwords.words('english'))\
    .union(stopwords.words('portuguese'))\
    .union(set(['.', '..', '...', ',', '?', '!']))

ALLOWED_ORIGINS = ['*']


class CorsMiddleware(object):

    def process_request(self, request, response):
        origin = request.get_header('Origin')
        if '*' in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
            response.set_header('Access-Control-Allow-Origin', origin)


class TagAutocompleteResource:

    def __init__(self, model):
        self.model = model

    def tokens(self, q):
        return TOKENIZER.tokenize(q)

    def most_similar(self, tokens, limit):
        return self.model.most_similar(positive=tokens, topn=limit)

    def lk(self, context):
        lk = []
        for w in context:
            if w.startswith("#") and w in self.model:
                lk.append(w)
            else:
                hw = '#'+w
                if hw in self.model and w in self.model:
                    c_w = self.model.vocab[w].count
                    c_hw = self.model.vocab[hw].count
                    logger.info(w + ': ' + str(c_w) + ' ' + hw + ': ' + str(c_hw))
                    if c_hw >= c_w:
                        lk.append(hw)
                    else:
                        lk.append(w)
                elif w in self.model:
                    lk.append(w)
        return lk

    def suggestions(self, q, limit):
        tokens = self.tokens(q)
        word = tokens[-1]
        context = tokens[:-1]
        context = filter(lambda x: x not in STOPWORDS, context)
        logger.info('word: ' + word + ' context: ' + ' '.join(context))
        lk = self.lk(context)
        logger.info('lk: ' + ' '.join(lk))
        most_similar = self.most_similar(lk, 5 * limit)
        return filter(lambda x: word in x[0], most_similar)

    def on_get(self, req, resp):
        q = req.get_param('q') or ''
        limit = req.get_param_as_int('limit') or 10

        try:
            suggestions = self.suggestions(q, limit)
            result = json.dumps([hit[0] for hit in suggestions])
        except Exception as ex:
            logger.error(ex)

            description = ('Aliens have attacked our base! We will '
                           'be back as soon as we fight them off. '
                           'We appreciate your patience.')

            raise falcon.HTTPServiceUnavailable(
                'Service Outage',
                description,
                30)

        resp.body = result

        resp.set_header('Powered-By', 'Falcon')
        resp.status = falcon.HTTP_200


# Useful for debugging problems in your API; works with pdb.set_trace(). You
# can also use Gunicorn to host your app. Gunicorn can be configured to
# auto-restart workers when it detects a code change, and it also works
# with pdb.


@plac.annotations(
    in_model=("Location of input model"),
    host=("Bind to host", "option", "b", str),
    port=("Bind to port", "option", "p", int),
)
def main(in_model, host='127.0.0.1', port=8001):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = Word2Vec.load(in_model)

    # Configure your WSGI server to load "quotes.app" (app is a WSGI callable)
    app = falcon.API(middleware=[
        CorsMiddleware()
    ])

    tag_autocomplete = TagAutocompleteResource(model)
    app.add_route('/', tag_autocomplete)

    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()


if __name__ == '__main__':
    plac.call(main)
