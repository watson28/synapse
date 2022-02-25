# Copyright 2016 OpenMarket Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import AsyncContextManager, Callable, Tuple

from twisted.internet import defer
from twisted.internet.defer import CancelledError, Deferred

from synapse.util.async_helpers import ReadWriteLock

from tests import unittest


class ReadWriteLockTestCase(unittest.TestCase):
    def _assert_called_before_not_after(self, lst, first_false):
        for i, d in enumerate(lst[:first_false]):
            self.assertTrue(d.called, msg="%d was unexpectedly false" % i)

        for i, d in enumerate(lst[first_false:]):
            self.assertFalse(
                d.called, msg="%d was unexpectedly true" % (i + first_false)
            )

    def test_rwlock(self):
        rwlock = ReadWriteLock()
        key = "key"

        def start_reader_or_writer(
            read_or_write: Callable[[str], AsyncContextManager]
        ) -> Tuple["Deferred[None]", "Deferred[None]"]:
            acquired_d: "Deferred[None]" = Deferred()
            release_d: "Deferred[None]" = Deferred()

            async def action():
                async with read_or_write(key):
                    acquired_d.callback(None)
                    await release_d

            defer.ensureDeferred(action())
            return acquired_d, release_d

        ds = [
            start_reader_or_writer(rwlock.read),  # 0
            start_reader_or_writer(rwlock.read),  # 1
            start_reader_or_writer(rwlock.write),  # 2
            start_reader_or_writer(rwlock.write),  # 3
            start_reader_or_writer(rwlock.read),  # 4
            start_reader_or_writer(rwlock.read),  # 5
            start_reader_or_writer(rwlock.write),  # 6
        ]
        # `Deferred`s that resolve when each reader or writer acquires the lock.
        acquired_ds = [acquired_d for acquired_d, _release_d in ds]
        # `Deferred`s that will trigger the release of locks when resolved.
        release_ds = [release_d for _acquired_d, release_d in ds]

        self._assert_called_before_not_after(acquired_ds, 2)

        self._assert_called_before_not_after(acquired_ds, 2)
        release_ds[0].callback(None)
        self._assert_called_before_not_after(acquired_ds, 2)

        self._assert_called_before_not_after(acquired_ds, 2)
        release_ds[1].callback(None)
        self._assert_called_before_not_after(acquired_ds, 3)

        self._assert_called_before_not_after(acquired_ds, 3)
        release_ds[2].callback(None)
        self._assert_called_before_not_after(acquired_ds, 4)

        self._assert_called_before_not_after(acquired_ds, 4)
        release_ds[3].callback(None)
        self._assert_called_before_not_after(acquired_ds, 6)

        self._assert_called_before_not_after(acquired_ds, 6)
        release_ds[5].callback(None)
        self._assert_called_before_not_after(acquired_ds, 6)

        self._assert_called_before_not_after(acquired_ds, 6)
        release_ds[4].callback(None)
        self._assert_called_before_not_after(acquired_ds, 7)

        release_ds[6].callback(None)

        acquired_d, release_d = start_reader_or_writer(rwlock.write)
        self.assertTrue(acquired_d.called)
        release_d.callback(None)

        acquired_d, release_d = start_reader_or_writer(rwlock.read)
        self.assertTrue(acquired_d.called)
        release_d.callback(None)

    def _start_reader(
        self, rwlock: ReadWriteLock, key: str, n: int
    ) -> Tuple["Deferred[None]", "Deferred[None]"]:
        """Starts a reader, which acquires the lock, blocks, then releases the lock."""
        unblock_d: "Deferred[None]" = Deferred()

        async def reader():
            async with rwlock.read(key):
                await unblock_d
            return f"read {n} completed"

        reader_d = defer.ensureDeferred(reader())
        return reader_d, unblock_d

    def _start_writer(
        self, rwlock: ReadWriteLock, key: str, n: int
    ) -> Tuple["Deferred[None]", "Deferred[None]"]:
        """Starts a writer, which acquires the lock, blocks, then releases the lock."""
        unblock_d: "Deferred[None]" = Deferred()

        async def writer():
            async with rwlock.write(key):
                await unblock_d
            return f"write {n} completed"

        writer_d = defer.ensureDeferred(writer())
        return writer_d, unblock_d

    def test_cancellation_with_read_lock(self):
        """Test cancellation while holding a read lock.

        A waiting writer should be given the lock when the reader holding the lock is
        cancelled.
        """
        rwlock = ReadWriteLock()
        key = "key"

        # 1. A reader takes the lock and blocks.
        reader_d, _ = self._start_reader(rwlock, key, 1)

        # 2. A writer waits for the reader to complete.
        writer_d, unblock_writer = self._start_writer(rwlock, key, 1)
        unblock_writer.callback(None)
        self.assertFalse(writer_d.called)

        # 3. The reader is cancelled.
        reader_d.cancel()
        self.failureResultOf(reader_d, CancelledError)

        # 4. The writer should take the lock and complete.
        self.assertTrue(
            writer_d.called, "Writer is stuck waiting for a cancelled reader"
        )
        self.assertEqual("write 1 completed", self.successResultOf(writer_d))

    def test_cancellation_with_write_lock(self):
        """Test cancellation while holding a write lock.

        A waiting reader should be given the lock when the writer holding the lock is
        cancelled.
        """
        rwlock = ReadWriteLock()
        key = "key"

        # 1. A writer takes the lock and blocks.
        writer_d, _ = self._start_writer(rwlock, key, 1)

        # 2. A reader waits for the writer to complete.
        reader_d, unblock_reader = self._start_reader(rwlock, key, 1)
        unblock_reader.callback(None)
        self.assertFalse(reader_d.called)

        # 3. The writer is cancelled.
        writer_d.cancel()
        self.failureResultOf(writer_d, CancelledError)

        # 4. The reader should take the lock and complete.
        self.assertTrue(
            reader_d.called, "Reader is stuck waiting for a cancelled writer"
        )
        self.assertEqual("read 1 completed", self.successResultOf(reader_d))

    def test_cancellation_waiting_for_read_lock(self):
        """Test cancellation while waiting for a read lock.

        Tests that cancelling a waiting reader:
         * does not cancel the writer it is waiting on
         * does not cancel the next writer waiting on it
         * does not allow the next writer to acquire the lock before an earlier writer
           has finished
         * does not keep the next writer waiting indefinitely
        """
        rwlock = ReadWriteLock()
        key = "key"

        # 1. A writer takes the lock and blocks.
        writer1_d, unblock_writer1 = self._start_writer(rwlock, key, 1)

        # 2. A reader waits for the first writer to complete.
        #    This reader will be cancelled later.
        reader_d, unblock_reader = self._start_reader(rwlock, key, 1)
        unblock_reader.callback(None)
        self.assertFalse(reader_d.called)

        # 3. A second writer waits for both the first writer and the reader to complete.
        writer2_d, unblock_writer2 = self._start_writer(rwlock, key, 2)
        unblock_writer2.callback(None)
        self.assertFalse(writer2_d.called)

        # 4. The waiting reader is cancelled.
        #    Neither of the writers should be cancelled.
        #    The second writer should still be waiting, but only on the first writer.
        reader_d.cancel()
        self.failureResultOf(reader_d, CancelledError)
        self.assertFalse(writer1_d.called, "First writer was unexpectedly cancelled")
        self.assertFalse(
            writer2_d.called,
            "Second writer was unexpectedly cancelled or given the lock before the "
            "first writer finished",
        )

        # 5. Unblock the first writer, which should complete.
        unblock_writer1.callback(None)
        self.assertEqual("write 1 completed", self.successResultOf(writer1_d))

        # 6. The second writer should take the lock and complete.
        self.assertTrue(
            writer2_d.called, "Second writer is stuck waiting for a cancelled reader"
        )
        self.assertEqual("write 2 completed", self.successResultOf(writer2_d))

    def test_cancellation_waiting_for_write_lock(self):
        """Test cancellation while waiting for a write lock.

        Tests that cancelling a waiting writer:
         * does not cancel the reader or writer it is waiting on
         * does not cancel the next writer waiting on it
         * does not allow the next writer to acquire the lock before an earlier reader
           and writer have finished
         * does not keep the next writer waiting indefinitely
        """
        rwlock = ReadWriteLock()
        key = "key"

        # 1. A reader takes the lock and blocks.
        reader_d, unblock_reader = self._start_reader(rwlock, key, 1)

        # 2. A writer waits for the reader to complete.
        writer1_d, unblock_writer1 = self._start_writer(rwlock, key, 1)

        # 3. A second writer waits for both the reader and first writer to complete.
        #    This writer will be cancelled later.
        writer2_d, unblock_writer2 = self._start_writer(rwlock, key, 2)
        unblock_writer2.callback(None)
        self.assertFalse(writer2_d.called)

        # 4. A third writer waits for the second writer to complete.
        writer3_d, unblock_writer3 = self._start_writer(rwlock, key, 3)
        unblock_writer3.callback(None)
        self.assertFalse(writer3_d.called)

        # 5. The second writer is cancelled.
        #    The reader, first writer and third writer should not be cancelled.
        #    The first writer should still be waiting on the reader.
        #    The third writer should still be waiting, even though the second writer has
        #    been cancelled.
        writer2_d.cancel()
        self.failureResultOf(writer2_d, CancelledError)
        self.assertFalse(reader_d.called, "Reader was unexpectedly cancelled")
        self.assertFalse(writer1_d.called, "First writer was unexpectedly cancelled")
        self.assertFalse(
            writer3_d.called,
            "Third writer was unexpectedly cancelled or given the lock before the first"
            "writer finished",
        )

        # 6. Unblock the reader, which should complete.
        #    The first writer should be given the lock and block.
        #    The third writer should still be waiting.
        unblock_reader.callback(None)
        self.assertEqual("read 1 completed", self.successResultOf(reader_d))
        self.assertFalse(
            writer3_d.called,
            "Third writer was unexpectedly given the lock before the first writer "
            "finished",
        )

        # 7. Unblock the first writer, which should complete.
        unblock_writer1.callback(None)
        self.assertEqual("write 1 completed", self.successResultOf(writer1_d))

        # 8. The third writer should take the lock and complete.
        self.assertTrue(
            writer3_d.called, "Third writer is stuck waiting for a cancelled writer"
        )
        self.assertEqual("write 3 completed", self.successResultOf(writer3_d))
