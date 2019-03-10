package com.example.periscopeai;

import android.app.ProgressDialog;
import android.hardware.Camera;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.speech.tts.TextToSpeech;
import android.support.v4.view.GestureDetectorCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.Spinner;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements GestureDetector.OnGestureListener, GestureDetector.OnDoubleTapListener {

    private static final String TAG = MainActivity.class.getName();
    Camera camera;
    private FrameLayout frameLayout;

    ShowCamera showCamera;

    String pic = "";

    private RequestQueue mRequestQueue;

    private StringRequest stringRequest;
    private String url = "http://d507e974.ngrok.io";
    private String image_processing_service = url + "/nik";

    private Spinner spinner;

    private MediaPlayer mediaPlayer;

    private Button play;
    private Button capture;

    private boolean playPause = false;
    private boolean initialStage = true;
    private ProgressDialog progressDialog;

    private TextToSpeech mTTS;

    private GestureDetectorCompat gestureDetector;

    private int languageIndex = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        spinner = (Spinner) findViewById(R.id.spinner);

        frameLayout = (FrameLayout) findViewById(R.id.frameLayout);

        play = (Button) findViewById(R.id.play);
        capture = (Button) findViewById(R.id.capture);

        ArrayAdapter<String> adapter = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_list_item_1, getResources().getStringArray(R.array.languages));
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        this.gestureDetector = new GestureDetectorCompat(this, this);
        this.gestureDetector.setOnDoubleTapListener(this);

        mTTS = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status == TextToSpeech.SUCCESS) {
                    int result = mTTS.setLanguage(Locale.GERMAN);

                    if(result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.e("TTS", "Language not supported.");
                    } else {
                        mTTS.setPitch(0.8f);
                        mTTS.setSpeechRate(0.99f);

                        mTTS.speak((String) spinner.getSelectedItem(), TextToSpeech.QUEUE_FLUSH, null);
                    }
                } else {
                    Log.e("TTS", "Initialization failed.");

                }
            }
        });

        progressDialog = new ProgressDialog(this);

        camera = Camera.open();
        showCamera = new ShowCamera(this, camera);

        frameLayout.addView(showCamera);

        mediaPlayer = new MediaPlayer();
        mediaPlayer.setAudioStreamType(AudioManager.STREAM_MUSIC);

        play.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!playPause) {
                    play.setText("Pause Streaming");

                    String speech = url + "/speech.mp3";

                    if (initialStage) {
                        new Player().execute(speech);
//                        new Player().execute("http://www.largesound.com/ashborytour/sound/brobob.mp3");
                    } else {
                        if (!mediaPlayer.isPlaying())
                            mediaPlayer.start();
                    }

                    playPause = true;

                } else {
                    play.setText("Launch Streaming");

                    if (mediaPlayer.isPlaying()) {
                        mediaPlayer.pause();
                    }

                    playPause = false;
                }
            }
        });
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        this.gestureDetector.onTouchEvent(e);

        return super.onTouchEvent(e);
    }

    @Override
    public boolean onSingleTapConfirmed(MotionEvent e) {


        return false;
    }

    @Override
    public boolean onDoubleTap(MotionEvent e) {

        System.out.println("double tap");
        startTimer();

        return false;
    }

    @Override
    public boolean onDoubleTapEvent(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onDown(MotionEvent e) {
        System.out.println("on down");
        return false;
    }

    @Override
    public void onShowPress(MotionEvent e) {

    }

    @Override
    public boolean onSingleTapUp(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
        return false;
    }

    @Override
    public void onLongPress(MotionEvent e) {

    }

    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {

        // System.out.println(velocityX + "\t" + velocityY);

        if(Math.abs(velocityX) > Math.abs(velocityY) && velocityX < 0) {
            System.out.println("Left");
        } else if(Math.abs(velocityX) > Math.abs(velocityY) && velocityX > 0) {
            System.out.println("Right");
        } else if(Math.abs(velocityX) < Math.abs(velocityY) && velocityX < 0) {
            System.out.println("Up ");

            languageIndex++;

            if(languageIndex > 9) {
                languageIndex = 0;
            }

            System.out.println(languageIndex);
            spinner.setSelection(languageIndex);
            mTTS.speak((String) spinner.getSelectedItem(), TextToSpeech.QUEUE_FLUSH, null);


        } else if(Math.abs(velocityX) < Math.abs(velocityY) && velocityX > 0) {
            System.out.println("Down " + languageIndex);

            languageIndex--;

            if(languageIndex < 0) {
                languageIndex = 9;
            }

            System.out.println(languageIndex);
            spinner.setSelection(languageIndex);
            mTTS.speak((String) spinner.getSelectedItem(), TextToSpeech.QUEUE_FLUSH, null);
        }

        return false;
    }

    public void startTimer() {
        new CountDownTimer(300000, 10000) {

            public void onTick(long millisUntilFinished) {
                captureImage();
            }

            public void onFinish() {
                System.out.println("Countdown ended.");
            }
        }.start();
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (mediaPlayer != null) {
            mediaPlayer.reset();
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }

    Camera.PictureCallback mPictureCallback = new Camera.PictureCallback() {

        @Override
        public void onPictureTaken(byte[] data, Camera camera) {
            System.out.println("Picture taken");

            pic = Base64.encodeToString(data, Base64.DEFAULT);

            sendRequestAndPrintResponse();

        }
    };

    private void sendRequestAndPrintResponse() {
        mRequestQueue = Volley.newRequestQueue(this);

        stringRequest = new StringRequest(Request.Method.POST, image_processing_service,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        Log.i(TAG, "Response: "  + response);

                        String speech = url + "/speech.mp3";
                        new Player().execute(speech);
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Log.i(TAG, "Error: " + error.toString());
                    }
                }
            ){
            @Override
            protected Map<String, String> getParams(){
                Map<String, String> params = new HashMap<>();


                params.put("message", "This message is from Nikandros.");
                params.put("pic", (pic.length() > 0) ? pic : "None");
                params.put("language", (String) spinner.getSelectedItem());

                camera.startPreview();

                return params;
            }

            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String,String> params = new HashMap<String, String>();
                params.put("Content-Type","application/x-www-form-urlencoded");
                return params;
            }
        };;

        mRequestQueue.add(stringRequest);
    }

    public void captureImage() {
        if(camera != null) {
            camera.takePicture(null, null, mPictureCallback);
        }
    }

    public void captureImage(View v) {
        /*if(camera != null) {
            camera.takePicture(null, null, mPictureCallback);
        }*/

        startTimer();
    }

    class Player extends AsyncTask<String, Void, Boolean> {
        @Override
        protected Boolean doInBackground(String... strings) {
            Boolean prepared = false;

            try {
                mediaPlayer.setDataSource(strings[0]);
                mediaPlayer.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                    @Override
                    public void onCompletion(MediaPlayer mediaPlayer) {
                        initialStage = true;
                        playPause = false;
                        play.setText("Launch Streaming");
                        mediaPlayer.stop();
                        mediaPlayer.reset();
                    }
                });

                mediaPlayer.prepare();
                prepared = true;

            } catch (Exception e) {
                Log.e("MyAudioStreamingApp", e.getMessage());
                prepared = false;
            }

            return prepared;
        }

        @Override
        protected void onPostExecute(Boolean aBoolean) {
            super.onPostExecute(aBoolean);

            if (progressDialog.isShowing()) {
                progressDialog.cancel();
            }

            mediaPlayer.start();
            initialStage = false;
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            progressDialog.setMessage("Buffering...");
            progressDialog.show();
        }
    }
}
