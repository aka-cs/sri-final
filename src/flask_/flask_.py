from pathlib import Path

from flask import Flask, render_template, send_from_directory, jsonify, request

from src.client.client import available_corpus, get_mri
from src.corpus.corpus import Document
from src.corpus.query import Query

app = Flask(__name__)

cfd = Path(__file__).parent.as_posix()


@app.route('/assets/<path:path>')
def send_js(path):
    return send_from_directory('assets', path)


@app.route('/api/corpus')
def send_corpus_list():
    return jsonify({k: v['name'] for k, v in available_corpus().items()})


@app.route('/api/corpus/<path:key>')
def send_query_answer(key):
    args = request.args
    if not args.get('q'):
        return "Record not found", 404
    page = args.get('page', '0')
    count = args.get('count', '10')
    if not page.isdigit() or not count.isdigit():
        return "Page and count must be integers", 400
    page, count = int(page), int(count)
    query = args['q']
    
    try:
        mri = get_mri(key, True)
    except KeyError:
        return 'Corpus not found', 404
    except Exception as e:
        raise e

    results: [Document, float] = mri.query(Query(query))
    
    paginated_results: [Document, float] = results[page * count:(page + 1) * count]
    paginated_results = [(doc.to_dict(), sim) for doc, sim in paginated_results]
    
    for result, _ in paginated_results:
        result['text'] = result['text'][:1000] + ('...' if len(result['text']) > 1000 else '')
    
    return jsonify({
        'page': page,
        'count': count,
        'total': len(results),
        'results': paginated_results,
    })


@app.route('/')
def index():
    return render_template('index.html')

