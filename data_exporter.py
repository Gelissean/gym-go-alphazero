import csv
import os
from multiprocessing import Lock


class data_exporter():
    def __init__(self, path):
        done = False
        temppath = path
        while not done:
            if os.path.exists(temppath):
                temppath = temppath[0:-3] + '1' + '.csv'
            else:
                done = True
        self.f = open(path, 'w')
        self.writer = csv.writer(self.f)
        self.lock = Lock()
        self.i_tasks = 0

    def __del__(self):
        if self.f is not None:
            self.f.close()

    def write(self, row):
        self.lock.acquire()
        self.writer.writerow(row)
        self.i_tasks = self.i_tasks + 1
        if self.i_tasks % 10 == 0:
            print("done another 10 tasks")
        self.lock.release()

    def close(self):
        self.f.close()
        self.f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f is not None:
            self.close()

def main():
    with data_exporter("subor.csv") as exporter:
        exporter.write(["udaj", "druhy","treti"])


if __name__ == '__main__':
    main()