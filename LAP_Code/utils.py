loggers = {}

def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger

def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)

def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state

def create_optimizer(optim_name, model, learning_rate=1e-3, weight_decay=0, **kwargs):
    """
    Creates an optimizer with the specified name and parameters.

    Args:
        optim_name (str): Name of the optimizer (e.g., 'Adam', 'SGD', 'AdamW').
        model (nn.Module): The model whose parameters will be optimized.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 0.
        **kwargs: Additional keyword arguments to be passed to the optimizer's constructor.

    Returns:
        torch.optim.Optimizer: The created optimizer instance.
    """

    if optim_name == 'Adadelta':
        rho = kwargs.get('rho', 0.9)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho, weight_decay=weight_decay)
    elif optim_name == 'Adagrad':
        lr_decay = kwargs.get('lr_decay', 0)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=lr_decay, weight_decay=weight_decay)
    elif optim_name == 'AdamW':
        betas = tuple(kwargs.get('betas', (0.9, 0.999)))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'SparseAdam':
        betas = tuple(kwargs.get('betas', (0.9, 0.999)))
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate, betas=betas)
    elif optim_name == 'Adamax':
        betas = tuple(kwargs.get('betas', (0.9, 0.999)))
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'ASGD':
        lambd = kwargs.get('lambd', 0.0001)
        alpha = kwargs.get('alpha', 0.75)
        t0 = kwargs.get('t0', 1e6)
        optimizer = optim.ASGD(model.parameters(), lr=learning_rate, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
    elif optim_name == 'LBFGS':
        max_iter = kwargs.get('max_iter', 20)
        max_eval = kwargs.get('max_eval', None)
        tolerance_grad = kwargs.get('tolerance_grad', 1e-7)
        tolerance_change = kwargs.get('tolerance_change', 1e-9)
        history_size = kwargs.get('history_size', 100)
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=max_iter, max_eval=max_eval,
                                tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
    elif optim_name == 'NAdam':
        betas = tuple(kwargs.get('betas', (0.9, 0.999)))
        momentum_decay = kwargs.get('momentum_decay', 4e-3)
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, betas=betas, momentum_decay=momentum_decay, weight_decay=weight_decay)
    elif optim_name == 'RAdam':
        betas = tuple(kwargs.get('betas', (0.9, 0.999)))
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'RMSprop':
        alpha = kwargs.get('alpha', 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay)
    elif optim_name == 'Rprop':
        momentum = kwargs.get('momentum', 0)
        optimizer = optim.Rprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optim_name == 'SGD':
        momentum = kwargs.get('momentum', 0)
        dampening = kwargs.get('dampening', 0)
        nesterov = kwargs.get('nesterov', False)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                              nesterov=nesterov, weight_decay=weight_decay)
    else:  # Adam is default
        betas = tuple(kwargs.get('betas', (0.9, 0.999)))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    return optimizer

class RunningAverage:
    """
    Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

def _split_and_move_to_gpu(t):
    def _move_to_gpu(input):
        if isinstance(input, (tuple, list)):
            return tuple([_move_to_gpu(x) for x in input])
        else:
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
            return input

    input, target = _move_to_gpu(t)
    return input, target

class TensorboardFormatter:
    """
    TensorboardFormatter 將 3D 體素數據 (NCDHW 格式) 轉換為 TensorBoard 可視化的影像格式，
    適用於自動編碼器 (Autoencoder) 進行 3D 重建，不包含語意分割相關處理。
    """

    def __init__(self, skip_last_target=False, log_channelwise=False):
        self.skip_last_target = skip_last_target
        self.log_channelwise = log_channelwise

    def __call__(self, name, batch):
        """
        轉換 batch 為 TensorBoard 可視化格式。

        Args:
            name (str): 'inputs' / 'targets' / 'predictions'
            batch (torch.Tensor): 4D (NDHW) 或 5D (NCDHW) 張量

        Returns:
            list[(str, np.ndarray)]: (標籤, 處理後影像) 的列表
        """
        def _check_img(tag_img):
            tag, img = tag_img
            assert img.ndim == 2 or img.ndim == 3, '僅支援 2D (HW) 或 3D (CHW) 圖像'
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)  # 轉為 (1, H, W)
            return tag, img

        tagged_images = self._process_batch(name, batch)
        return list(map(_check_img, tagged_images))

    def _process_batch(self, name, batch):
        """
        取 3D 體素的中間切片作為 2D 影像顯示。
        """
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/slice_{}'
        tagged_images = []

        if batch.ndim == 6:  # (2, 558, 256, 3, 3, 3)
            D = batch.shape[3]
            slice_idx = D // 2 if D % 2 == 0 else (D // 2 + 1)  # 確保中心切片正確
            for batch_idx in range(batch.shape[0]):  # 兩隻蟲
                for cell_idx in range(batch.shape[1]):  # 558 個細胞核
                    tag = tag_template.format(name, batch_idx, cell_idx)
                    img = batch[batch_idx, cell_idx, :, slice_idx, :, :].mean(axis=0)  # 平均 256 channels 再轉成 2D
                    tagged_images.append((tag, self._normalize_img(img)))

        elif batch.ndim == 5:  # NCDHW
                slice_idx = batch.shape[2] // 2  # 取中間切片
                for batch_idx in range(batch.shape[0]):
                    tag = tag_template.format(name, batch_idx, slice_idx)
                    img = batch[batch_idx, 0, slice_idx, ...]  # 取 C=1 的影像
                    tagged_images.append((tag, self._normalize_img(img)))

        else:  # NDHW 格式
            slice_idx = batch.shape[1] // 2
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        """ 影像歸一化至 [0, 1] 範圍 """
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))