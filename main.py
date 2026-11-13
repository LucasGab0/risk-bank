"""Entrada do projeto: pesquisa completa, comparação rápida ou servidor local."""
import sys


def main():
    if '--serve' in sys.argv or '-s' in sys.argv:
        import uvicorn
        uvicorn.run('server:app', host='127.0.0.1', port=8000)
    elif '--baseline' in sys.argv:
        sys.argv.remove('--baseline')
        from train import main as train_main
        train_main()
    else:
        if '--research' in sys.argv:
            sys.argv.remove('--research')
        try:
            from research import main as research_main
        except ModuleNotFoundError as error:
            if error.name == 'torch':
                raise SystemExit('Para executar a pesquisa, instale requirements-research.txt. Para abrir o aplicativo: python main.py --serve.') from None
            raise
        if '--integrate' not in sys.argv:
            sys.argv.append('--integrate')
        if '--resume' not in sys.argv:
            sys.argv.append('--resume')
        research_main()


if __name__ == '__main__':
    main()
