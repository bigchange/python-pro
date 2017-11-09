
# refer ：https://github.com/google/python-fire
import fire


class IngestionStage(object):

    def run(self):
        return 'Ingesting! Nom nom nom...'


class DigestionStage(object):

    def run(self, volume=1):
        return ' '.join(['Burp!'] * volume)

    def status(self):
        return 'Satiated.'


class Pipeline(object):

    def __init__(self):
        self.ingestion = IngestionStage()
        self.digestion = DigestionStage()

    def run(self):
        self.ingestion.run()
        self.digestion.run()


class Example(object):
    def hello(self, name='world'):
        """Says hello to the specified name."""
        return 'Hello {name}!'.format(name=name)


# python hello_world.py Example hello  --name="youcj"
# python example.py ingestion run

def main():
    fire.Fire()
    # we can try to use: fire.Fire(Example) to see the diff in CLI


if __name__ == '__main__':
    main()
