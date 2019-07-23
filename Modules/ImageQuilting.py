import numpy as np
from numpy.lib.stride_tricks import as_strided
import sys


class ImageQuilting:
    def __init__(self, inputTexture):
        self.inputTexture = inputTexture
        self.shorterEdge = min(inputTexture.shape[0], inputTexture.shape[1])
        self.passValidation = True
        self.tolerance = 1.05

    def __call__(self, outputSize, blockSize, overlapEdge=None, randomSeed=0):
        if not self.isInputValid(outputSize, blockSize, overlapEdge):
            print('input validation failed')
            return

        N = self.blockSize
        ov = self.overlapEdge

        np.random.seed(randomSeed)

        viewSize = (
            self.inputTexture.shape[0] - N + 1,
            self.inputTexture.shape[1] - N + 1,
            N,
            N,
            self.inputTexture.shape[2]
        )
        strides = self.inputTexture.strides[:2] + self.inputTexture.strides
        blockView = as_strided(self.inputTexture, viewSize, strides)
        blockAmount = blockView.shape[0] * blockView.shape[1]
        blockView = blockView.reshape(
            blockAmount,
            blockView.shape[2],
            blockView.shape[3],
            blockView.shape[4]
        )  # (blockAmount, blockSize, blockSize, ch)

        # rasterscan (blockの左上点)
        rowRange = np.arange(
            0, self.outputSize[0], N - self.overlapEdge)
        colRange = np.arange(
            0, self.outputSize[1], N - self.overlapEdge)

        for (i, y) in enumerate(rowRange):
            rowInfo = str(i + 1) + '/' + str(len(rowRange)) + ':'
            for (j, x) in enumerate(colRange):
                colProgress = '[' + '|' * j + '.' * (len(colRange) - j - 1) + ']'
                sys.stdout.write("\033[2K\033[G%s" % (rowInfo + colProgress))
                sys.stdout.flush()

                if (y == 0) & (x == 0):  # 左上
                    B = blockView[np.random.randint(0, blockView.shape[0])]
                    self.output[0:N, 0:N] = B

                elif y == 0:  # 上端なので左端のみ切る
                    Eset = (blockView[:, :, 0:ov, :] - self.output[None, y:(y + N), x:(x + ov), :]) ** 2
                    Eset = Eset.sum(axis=3)  # 画素単位の誤差
                    Esum = Eset.sum(axis=(1, 2))  # 交差領域全体の誤差
                    EminIndices = np.arange(blockAmount)[(Esum <= (Esum.min() * self.tolerance))]
                    ix = np.random.choice(EminIndices)
                    Ecum, path = self.findPath(Eset[ix])
                    patch, patchline = self.cutPatchV(blockView[ix], Ecum, path, y, x)
                    self.output[y:(y + N), x:(x + N)] = patch
                    self.patchLine[y:(y + N), x:(x + N)] = patchline

                elif x == 0:  # 左端なので上端のみ切る
                    Eset = (blockView[:, 0:ov, :, :] - self.output[None, y:(y + ov), x:(x + N), :]) ** 2
                    Eset = Eset.sum(axis=3)  # 画素単位の誤差
                    Esum = Eset.sum(axis=(1, 2))  # 交差領域全体の誤差
                    EminIndices = np.arange(blockAmount)[(Esum <= (Esum.min() * self.tolerance))]
                    ix = np.random.choice(EminIndices)
                    Ecum, path = self.findPath(Eset[ix].T)
                    patch, patchline = self.cutPatchH(blockView[ix], Ecum, path, y, x)
                    self.output[y:(y + N), x:(x + N)] = patch
                    self.patchLine[y:(y + N), x:(x + N)] = patchline

                else:  # それ以外 左と上
                    EsetV = (blockView[:, :, 0:ov, :] - self.output[None, y:(y + N), x:(x + ov), :]) ** 2
                    EsetH = (blockView[:, 0:ov, :, :] - self.output[None, y:(y + ov), x:(x + N), :]) ** 2
                    EsetHV = (blockView[:, 0:ov, 0:ov, :] - self.output[None, y:(y + ov), x:(x + ov), :]) ** 2
                    EsetV = EsetV.sum(axis=3)
                    EsetH = EsetH.sum(axis=3)
                    EsetHV = EsetHV.sum(axis=3)
                    Esum = EsetV.sum(axis=(1, 2)) + EsetH.sum(axis=(1, 2)) - EsetHV.sum(axis=(1, 2))
                    EminIndices = np.arange(blockAmount)[(Esum <= (Esum.min() * self.tolerance))]
                    ix = np.random.choice(EminIndices)

                    # 縦に切って横に切る
                    Ecum, path = self.findPath(EsetV[ix])
                    patch, patchlineV = self.cutPatchV(blockView[ix], Ecum, path, y, x)
                    Ecum, path = self.findPath(EsetH[ix].T)
                    patch, patchlineH = self.cutPatchH(patch, Ecum, path, y, x)
                    self.output[y:(y + N), x:(x + N)] = patch
                    self.patchLine[y:(y + N), x:(x + N)] = patchlineV + patchlineH
        print()
        self.output = self.output.astype('uint8')[:self.outputSize[0], :self.outputSize[1]]
        self.patchLine = self.patchLine.astype('uint8')[:self.outputSize[0], :self.outputSize[1]]

    def findPath(self, E):  # E:縦長のoverlap領域に上から下にパスを下ろす
        path = np.zeros(E.shape, dtype='int')  # min path to [i,j] in {↘,↓,↙}
        Ecum = np.copy(E)
        row, col = E.shape

        for i in range(1, row):
            for j in range(col):
                l = max(0, j - 1)
                r = min(col, j + 2)
                Ecum[i, j] = E[i, j] + np.min(E[i - 1, l:r])
                path[i, j] = np.argmin(E[i - 1, l:r]) + l

        return Ecum, path

    def cutPatchV(self, patch, Ecum, path, y, x):
        resultPatch = np.copy(patch)
        patchline = np.zeros(patch.shape)
        ptr = np.argmin(Ecum[-1, :])
        for i in range(len(Ecum))[::-1]:
            resultPatch[i, ptr] = (resultPatch[i, ptr] + self.output[y + i, x + ptr]) / 2
            resultPatch[i, :ptr] = self.output[y + i, x:(x + ptr)]
            patchline[i, ptr] = np.array([0, 0, 255])
            ptr = path[i, ptr]
        return resultPatch, patchline

    def cutPatchH(self, patch, Ecum, path, y, x):
        resultPatch = np.copy(patch)
        patchline = np.zeros(patch.shape)
        ptr = np.argmin(Ecum[-1, :])
        for i in range(len(Ecum))[::-1]:
            resultPatch[:ptr, i] = (resultPatch[ptr, i] + self.output[y + ptr, x + i]) / 2
            resultPatch[:ptr, i] = self.output[y:(y + ptr), x + i]
            patchline[ptr, i] = np.array([0, 0, 255])
            ptr = path[i, ptr]
        return resultPatch, patchline

    def isInputValid(self, outputSize, blockSize, overlapEdge):
        if type(outputSize) == int:
            outputSize = (outputSize, outputSize)
        elif not (
                type(outputSize) == tuple
                & len(outputSize) == 2
                & type(outputSize[0]) == int
                & type(outputSize[1]) == int
        ):
            print('outputSize must be integer or tuple(int, int)')
            self.passValidation = False

        if not (type(blockSize) == int) & (
                (overlapEdge is None) | (type(overlapEdge) == int)):
            print('blockSize, and overlapEdge(if given) must be integer')
            self.passValidation = False

        if self.shorterEdge <= blockSize:
            print(
                'blockSize is too large, must be less than shorterEdge of inputTexture:', self.shorterEdge)
            self.passValidation = False

        if min(outputSize) <= blockSize:
            print('outputSize <= blockSize is meaningless')
            self.passValidation = False

        if (overlapEdge is not None):
            if (overlapEdge >= blockSize):
                print(
                    'overlapEdge is too large, must be less than blockSize:', blockSize)
                self.passValidation = False
        else:
            overlapEdge = blockSize // 6

        self.outputSize, self.blockSize, self.overlapEdge = (
            outputSize, blockSize, overlapEdge)
        self.output = np.zeros((outputSize[0] + self.blockSize, outputSize[1] + self.blockSize, 3), dtype='int')
        self.patchLine = np.copy(self.output)

        return self.passValidation
