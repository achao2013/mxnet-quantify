package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.{Logger, LoggerFactory}

/**
 * Server node for the key value store
 * @author Yizhi Liu
 */
class KVStoreServer(private val kvStore: KVStore) {
  private val logger: Logger = LoggerFactory.getLogger(classOf[KVStoreServer])
  private val handle: KVStoreHandle = kvStore.handle
  private val controller = new KVServerControllerCallback {
    override def invoke(cmdId: Int, cmdBody: String): Unit = {
      logger.debug("Receive cmdId {}, cmdBody: {}", cmdId, cmdBody)
      if (cmdId == 0) {
        val optimizer = Serializer.getSerializer.deserialize[Optimizer](
          Serializer.decodeBase64String(cmdBody))
        kvStore.setOptimizer(optimizer)
      } else {
        logger.warn(s"Server ${kvStore.rank}, unknown command ($cmdId, $cmdBody)")
      }
    }
  }

  // run the server, whose behavior is like
  // while receive(x):
  //   if is_command x: controller(x)
  //   else if is_key_value x: updater(x)
  def run(): Unit = {
    checkCall(_LIB.mxKVStoreRunServer(handle, controller))
  }
}

object KVStoreServer {
  // Start server/scheduler according to env variables
  def start(): Unit = {
    val isWorker = new RefInt
    checkCall(_LIB.mxKVStoreIsWorkerNode(isWorker))
    require(isWorker.value == 0, "cannot start kv-store server on worker node")
    val kvStore = KVStore.create("dist")
    val server = new KVStoreServer(kvStore)
    server.run()
  }

  def init(env: Map[String, String]): Unit = {
    val keys = env.keys.toArray
    val vals = env.values.toArray
    checkCall(_LIB.mxInitPSEnv(keys, vals))
  }
}

trait KVServerControllerCallback {
  def invoke(cmdId: Int, cmdBody: String): Unit
}
