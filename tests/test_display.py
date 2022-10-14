# class TestDisplay:
#     def __enter__(self, )


class Batch:
    def __init__(self, dataloader=range(10)):
        self.data = dataloader

    def __iter__(self):
        print("start")
        for val in self.data:
            yield val

    # def __next__(self):
    #     print("hey")
    #     # return next(self.data)
    #     yield next(self.data)


b = Batch()

for i in b:
    print(i)
