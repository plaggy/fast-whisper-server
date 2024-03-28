import http from 'k6/http';
import { Trend } from 'k6/metrics';

const shortAssistedTime = new Trend('short_assisted', true);
const shortNotAssistedTime = new Trend('short_not_assisted', true);
const longAssistedTime = new Trend('long_assisted', true);
const longNotAssistedTime = new Trend('long_not_assisted', true);

const url = 'http://0.0.0.0:7860';

const audios = {
    'short': open('model-server/app/tests/polyai-minds14-0.wav', 'rb'), // 8s
    'long': open('ted_60.wav', 'rb') // 60s
};

export const options = {
    scenarios: {
        short_audio_not_assisted: {
            executor: 'constant-vus',
            vus: 1,
            startTime: '0s',
            duration: '30s',
            env: { 
                AUDIO: 'short',
                BATCH_SIZE: '24',
                ASSISTED: 'false'
            },
        },
        short_audio_assisted: {
            executor: 'constant-vus',
            vus: 1,
            startTime: '30s',
            duration: '30s',
            env: { 
                AUDIO: 'short',
                BATCH_SIZE: '1',
                ASSISTED: 'true'
            },
        },
        long_audio_not_assisted: {
            executor: 'constant-vus',
            vus: 1,
            startTime: '1m',
            duration: '1m',
            env: { 
                AUDIO: 'long',
                BATCH_SIZE: '24',
                ASSISTED: 'false'
            },
        },
        long_audio_assisted: {
            executor: 'constant-vus',
            vus: 1,
            startTime: '2m',
            duration: '1m',
            env: { 
                AUDIO: 'long',
                BATCH_SIZE: '1',
                ASSISTED: 'true'
            },
        },
    },
};


export default function() {
    let parameters = JSON.stringify({'batch_size': __ENV.BATCH_SIZE, 'assisted': __ENV.ASSISTED});
    const data = {
        parameters: parameters,
        file: http.file(audios[__ENV.AUDIO], 'filename', 'audio/wav'),
    };

    const resp = http.post(url, data);
    if (__ENV.AUDIO == 'short' && __ENV.ASSISTED == 'false'){
        shortNotAssistedTime.add(resp.timings.duration);
    } else if (__ENV.AUDIO == 'short' && __ENV.ASSISTED == 'true'){
        shortAssistedTime.add(resp.timings.duration);
    } else if (__ENV.AUDIO == 'long' && __ENV.ASSISTED == 'false'){
        longNotAssistedTime.add(resp.timings.duration);
    } else if (__ENV.AUDIO == 'long' && __ENV.ASSISTED == 'true'){
        longAssistedTime.add(resp.timings.duration);
    }
    console.log(resp.status);
};