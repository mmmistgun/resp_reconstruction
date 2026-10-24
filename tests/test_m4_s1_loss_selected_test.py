"""checkpoint总loss选点的合成边界测试。"""
import pytest
from scripts.eval_m4_s1_loss_selected_test import select_loss


def row(epoch, loss, rr=1., train=.1):
    return {'epoch':epoch,'validation':{'loss':loss},'val_local_rr_mae':rr,'train':{'loss':train}}


def test_total_validation_loss_not_training_or_rr():
    history=[row(1,.5,.1,.01),row(2,.3,2.,.9),row(3,.4,.05,.001)]
    assert select_loss(history)['epoch']==2


def test_earliest_tie_and_full_history():
    assert select_loss([row(1,.5),row(2,.3),row(3,.3)])['epoch']==2
    assert select_loss([row(1,.5),row(2,.3),row(3,.2)])['epoch']==3


@pytest.mark.parametrize('history',[[],[row(2,.1)],[row(1,.2),row(3,.1)],[row(1,.2),row(1,.1)]])
def test_incomplete_history_fails(history):
    with pytest.raises(ValueError):select_loss(history)


@pytest.mark.parametrize('loss',[float('nan'),float('inf'),None,True])
def test_nonfinite_or_missing_loss_fails(loss):
    with pytest.raises(ValueError):select_loss([row(1,.2),row(2,loss)])
