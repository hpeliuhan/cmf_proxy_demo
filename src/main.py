from waggle.plugin import Plugin
import time
from waggle.time import get_timestamp

def main():
    with Plugin() as plugin:
        t=time.time()
        plugin.publish("helloworld", t)
        plugin.upload_file("test.txt",timestamp=get_timestamp())
if __name__ == "__main__":
    main()