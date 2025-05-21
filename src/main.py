from waggle.plugin import Plugin
import time

def main():
    with Plugin() as plugin:
        t=time.time()
        plugin.publish("helloworld", t)
        plugin.upload_file("test.txt", t)

if __name__ == "__main__":
    main()