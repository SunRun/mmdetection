from .coco import CocoDataset
from .builder import DATASETS

import os.path as osp
from pycocotools.cocoeval import COCOeval
import logging
import numpy as np
import time, datetime
import copy


class FacetsEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
        super().__init__(cocoGt, cocoDt, iouType)
        self.values = {}

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))

            return mean_s, s[s > -1]

        if not self.eval:
            raise Exception("Please run accumulate() first")

        stats = np.zeros((12,))
        stats[0], self.values["AP_all_0.5-0.95"] = _summarize(1)
        stats[1], self.values["AP_all_0.5"] = _summarize(
            1, iouThr=0.5, maxDets=self.params.maxDets[2]
        )
        stats[2], self.values["AP_all_0.75"] = _summarize(
            1, iouThr=0.75, maxDets=self.params.maxDets[2]
        )
        stats[3], self.values["AP_small_0.5-0.95"] = _summarize(
            1, areaRng="small", maxDets=self.params.maxDets[2]
        )
        stats[4], self.values["AP_medium_0.5-0.95"] = _summarize(
            1, areaRng="medium", maxDets=self.params.maxDets[2]
        )
        stats[5], self.values["AP_large_0.5-0.95"] = _summarize(
            1, areaRng="large", maxDets=self.params.maxDets[2]
        )
        stats[6], self.values["AR_all_0.5-0.95_max1"] = _summarize(
            0, maxDets=self.params.maxDets[0]
        )
        stats[7], self.values["AR_all_0.5-0.95_max10"] = _summarize(
            0, maxDets=self.params.maxDets[1]
        )
        stats[8], self.values["AR_all_0.5-0.95_max100"] = _summarize(
            0, maxDets=self.params.maxDets[2]
        )
        stats[9], self.values["AR_small_0.5-0.95"] = _summarize(
            0, areaRng="small", maxDets=self.params.maxDets[2]
        )
        stats[10], self.values["AR_medium_0.5-0.95"] = _summarize(
            0, areaRng="medium", maxDets=self.params.maxDets[2]
        )
        stats[11], self.values["AR_large_0.5-0.95"] = _summarize(
            0, areaRng="large", maxDets=self.params.maxDets[2]
        )

        self.stats = stats

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        # print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            # print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        # print('DONE (t={:0.2f}s).'.format(toc-tic))

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        # print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            # print('Please run evaluate() first')
            pass
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones(
            (T, R, K, A, M)
        )  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e["dtMatches"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    dtIg = np.concatenate(
                        [e["dtIgnore"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        toc = time.time()
        # print('DONE (t={:0.2f}s).'.format(toc - tic))

    def __str__(self):
        self.summarize()


@DATASETS.register_module()
class FacetsDataset(CocoDataset):
    CLASSES = "facet"

    def __init__(self, dsm_prefix=None, *args, **kwargs):
        super(FacetsDataset, self).__init__(*args, **kwargs)

        self.dsm_prefix = dsm_prefix
        if not (self.dsm_prefix is None or osp.isabs(self.dsm_prefix)):
            self.dsm_prefix = osp.join(self.data_root, self.dsm_prefix)

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results["dsm_prefix"] = self.dsm_prefix

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=np.arange(0.5, 0.96, 0.05),
    ):
        """Evaluation in COCO protocol.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.
        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ["bbox", "segm", "proposal", "proposal_fast"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric {} is not supported".format(metric))

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = "Evaluating {}...".format(metric)
            if logger is None:
                msg = "\n" + msg
            # print_log(msg, logger=logger)

            if metric == "proposal_fast":
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger="silent"
                )
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results["AR@{}".format(num)] = ar[i]
                    log_msg.append("\nAR@{}\t{:.4f}".format(num, ar[i]))
                log_msg = "".join(log_msg)
                # print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError("{} is not in results".format(metric))
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                # print_log(
                #     'The testing results of the whole dataset is empty.',
                #     logger=logger,
                #     level=logging.ERROR)
                break

            iou_type = "bbox" if metric == "proposal" else metric
            cocoEval = FacetsEval(cocoGt, cocoDt, iou_type)  # COCOeval -> FacetsEval
            cocoEval.params.imgIds = self.img_ids
            if metric == "proposal":
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    "AR@100",
                    "AR@300",
                    "AR@1000",
                    "AR_s@1000",
                    "AR_m@1000",
                    "AR_l@1000",
                ]
                for i, item in enumerate(metric_items):
                    val = float("{:.3f}".format(cocoEval.stats[i + 6]))
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    pass  # TODO
                metric_items = ["mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]
                for i in range(len(metric_items)):
                    key = "{}_{}".format(metric, metric_items[i])
                    val = float("{:.3f}".format(cocoEval.stats[i]))
                    eval_results[key] = val
                eval_results["{}_mAP_copypaste".format(metric)] = (
                    "{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} "
                    "{ap[4]:.3f} {ap[5]:.3f}"
                ).format(ap=cocoEval.stats[:6])
            vls = cocoEval.values
            imgs = cocoEval.evalImgs
            im_eval = cocoEval.eval
            ious = cocoEval.ious

        other_res = {"stats": vls, "imgs": imgs, "ious": ious}

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results, other_res
