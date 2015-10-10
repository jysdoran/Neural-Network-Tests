using UnityEngine;
using System.Collections;

public class NeuralNet : MonoBehaviour {

    public float[,] synone;
    public float[,] syntwo;
    public int n;

    public float rate;

    public float[] sigma;
    public float[] sigmoid;

    public float[] medin;
    public float[] medout;

    public float[] input;
    public float[] output;
    public float[] actual;
    public float[] error;

	// Use this for initialization
	void Start () {
        n = 30;
        rate = 0.1f;

        synone = new float[n, n];
        syntwo = new float[n, n];

        sigma = new float[n];
        sigmoid = new float[n];
        medin = new float[n];
        medout = new float[n];
        input = new float[2];
        output = new float[2];
        error = new float[2];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                synone[i,j] = 0.1f * Random.value;
                syntwo[i,j] = 0.1f * Random.value;
            }
        }

	}
	
	// Update is called once per frame
	void Update () {
        input[0] = 1f;
        input[1] = Random.value * Mathf.PI * 2;
        actual = ConvertPolar(input[0], input[1]);
        for (int i = 0; i < n; i++)
        {
            medin[i] = 0;
            for (int j = 0; j < 2; j++)
            {
                medin[i] += synone[j,i] * input[j];
                if (float.IsNaN(medin[i]))
                {
                    Debug.LogError(synone[j, i]);
                } else if (medin[i] > 100f)
                {
                    Debug.Log(j + ", " + i + " : " + medin[i]);
                    Debug.Log("synone: " + synone[j, i]);
                    Debug.Log("syntwo: " + syntwo[i, j]);
                    Debug.Log("sigma: " + sigma[j]);
                    Debug.Log("sigmoid: " + sigmoid[j]);
                }
            }
            medout[i] = tanh(medin[i]);
        }
        for (int i = 0; i < 2; i++)
        {
            output[i] = 0;
            for (int j = 0; j < n; j++)
            {
                output[i] += syntwo[j,i] * medout[j];
                if (float.IsNaN(output[i]))
                {
                    Debug.Log(syntwo[j,i]);
                    Debug.Log(medout[j]);
                }
            }
            error[i] = actual[i] - output[i];
        }
        //Back Propagation
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < n; j++)
            {
                syntwo[j,i] += rate * medout[j] * error[i];
            }
        }
        for (int i = 0; i < n; i++)
        {
            sigma[i] = 0;
            for (int j = 0; j < 2; j++)
            {
                sigma[i] += error[j] * syntwo[i,j];
            }
            sigmoid[i] = 1 - medin[i] * medin[i];
        }
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < n; j++)
            {
                synone[i,j] += rate * sigmoid[j] * sigma[j] * input[i];
            }
        }
        //Debug.Log(Mathf.Sqrt(error[0] * error[0] + error[1] * error[1]));
      //  Debug.Log(output[0]);
    }

    float tanh (float x)
    {
        if (float.IsNaN(x))
        {
            //Debug.Log(x);
            x = 5f;
        }
        if (x > 5f)
        {
            x = 5f;
        } else if (x < -5f)
        {
            x = -5f;
        }
        float output = (Mathf.Exp(x) - Mathf.Exp(-x)) / (Mathf.Exp(x) + Mathf.Exp(-x));
        if (float.IsNaN(output))
        {
            Debug.Log(x);
        }
        return output;
    }

    float[] ConvertPolar(float r, float theta)
    {
        float[] output = new float[2];

        output[0] = r * Mathf.Cos(theta); //x
        output[1] = r * Mathf.Sin(theta); //y

        return output;
    }


}
