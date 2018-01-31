#
# parallel.py
# wrapper for easy scripting of MPI-parallelized tasks
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

from mpi4py import MPI
import sys
import math
import time

def info():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    master = ( rank == 0 )
    return comm, size, rank, master

def partition(l):
    comm, size, rank, master = info()
    division = len(l) / float(size)
    return [ l[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(size) ][rank]

class ETA:
    def __init__(self,n=0,reports=1):
        # record start time
        self.startTime = time.time()
        # calculate reporting interval
        self.n = n
        self.interval = int(math.ceil(float(n)/reports))
        if self.interval < 1:
            self.interval = 1

    def report(self,i):
        if i > 0 and i%self.interval == 0:
            elapsed = time.time()-self.startTime
            estimate = elapsed * self.n / float(i)
            print('Completed %d of %d (%.0f percent) : %.1f s of %.1f s'%
                  (i,self.n,float(i)/self.n*100.,elapsed,estimate))
            sys.stdout.flush()

class ParallelTask:
    def __init__(self):
        self.comm, self.size, self.rank, self.master = info()
        self.data = None
        self.solo_mode = False
        if self.master:
                print('ParallelTask initialized with %d MPI ranks'%self.size)
        if self.size < 2:
            print('Warning: ParallelTask is only aware of one MPI rank! Computing in solo mode.')
            self.solo_mode = True

    def shareData(self,data):
        if self.master:
            self.data = data
            for i in range(1,self.size):
                self.comm.send(data, dest=i, tag=0)
        else:
            sys.stdout.flush()
            self.data = self.comm.recv(source=0, tag=0)
        return self.data

    def gatherData(self,data):
        if self.master:
            datas = []
            for i in range(1,self.size):
                buff = self.comm.recv(source=i, tag=0)
                datas.append(buff)
            return datas
        else:
            self.comm.send(data, dest=0, tag=0)
            return None

    def computeQueue(self,function=None,tasks=None,reports=10):
        if self.solo_mode:
            return self.soloCompute(function,tasks,reports)
        if self.master:
            return self.masterCompute(tasks,reports)
        else:
            return self.slaveCompute(function)

    def slaveCompute(self,function):
        while True:
            # await instructions
            (index, task) = self.comm.recv(source=0, tag=1)
            if task is None:
                break
            # perform the task, supplying data synced across ranks
            result = function(task,self.data)
            # send back to master
            buff = (self.rank, index, task, result)
            self.comm.send(buff, dest=0, tag=2)
        return None

    def masterCompute(self,tasks,reports):
        # set up accounting system for tasks
        nThreads = min(self.size,len(tasks))
        busyThreads = []
        idleThreads = [ x for x in range(1,nThreads) if x not in busyThreads ]
        sendCount = 0
        recvCount = 0
        countdown = ETA(len(tasks),reports)
        # release unecessary ranks right away
        for i in range(len(tasks),self.size):
            self.comm.send((None,None), dest=i, tag=1)
        # run through the queue
        results = [i for i in range(len(tasks))]
        while recvCount < len(tasks):
            while ( len(idleThreads) > 0 ) and ( sendCount < len(tasks) ):
                # send tasks to idle threads
                dest = idleThreads.pop()
                buff = (sendCount,tasks[sendCount])
                self.comm.send(buff, dest=dest, tag=1)
                busyThreads.append(dest)
                sendCount += 1
            # receive completed tasks
            buff = self.comm.recv(source=MPI.ANY_SOURCE, tag=2)
            rank, index, task, result = buff
            recvCount += 1
            countdown.report(recvCount)
            results[index] = result
            busyThreads.remove(rank)
            # release workers when tasks are done
            if sendCount == len(tasks):
                self.comm.send((None,None), dest=rank, tag=1)
            else:
                idleThreads.append(rank)
        # report elapsed time
        elapsed = time.time()-countdown.startTime
        print('Parallel queue complete. Execution time: ',elapsed)
        # return result
        return results

    def soloCompute(self,function,tasks,reports):
        countdown = ETA(len(tasks),reports)
        results = []
        for i in range(len(tasks)):
            results.append( function(tasks[i], self.data ) )
            countdown.report(i)
        # report elapsed time
        elapsed = time.time()-countdown.startTime
        print('Solo queue complete. Execution time: ',elapsed)
        # return result
        return results
