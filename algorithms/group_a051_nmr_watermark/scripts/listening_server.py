"""Local webMUSHRA host with a Python result collector; no PHP required."""
import argparse
import json
import uuid
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import uvicorn


def build_server(study, vendor):
    study, vendor = Path(study).resolve(), Path(vendor).resolve()
    app = FastAPI()

    @app.post('/service/write.php')
    async def save_response(request: Request):
        body = (await request.body()).decode('utf-8')
        # webMUSHRA sends raw JSON after sessionJSON= (not URL encoded).
        if not body.startswith('sessionJSON='):
            raise HTTPException(400, 'Missing sessionJSON')
        try:
            value = json.loads(body[len('sessionJSON='):])
        except ValueError:
            raise HTTPException(400, 'Invalid session JSON')
        if not isinstance(value, dict) or value.get('testId') != 'a051' or not isinstance(value.get('trials'), list):
            raise HTTPException(400, 'Unexpected experiment')
        folder = study/'responses'
        folder.mkdir(exist_ok=True)
        (folder/f'{uuid.uuid4().hex}.json').write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return Response(content='', media_type='text/plain')

    @app.get('/configs/a051.yaml')
    def config():
        return Response((study/'a051.yaml').read_text(encoding='utf-8'), media_type='text/yaml')

    app.mount('/study/audio', StaticFiles(directory=study/'audio'), name='audio')
    # Do not expose private model mapping or participant responses as static files.
    app.mount('/', StaticFiles(directory=vendor, html=True), name='webmushra')
    return app


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', default='outputs/listening')
    p.add_argument('--vendor', default='vendor/webMUSHRA')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(build_server(args.study, args.vendor), host='127.0.0.1', port=args.port)
