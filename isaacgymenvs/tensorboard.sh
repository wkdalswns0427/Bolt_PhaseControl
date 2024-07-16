#!bin/sh
echo "run tensorboard on /runs"
if python -c "tensorboard" &> /dev/null; then
    echo 'tensorboard exists running...'
else
    echo 'installing tensorboard'
    pip3 install tensorboard
fi

tensorboard --logdir=runs
